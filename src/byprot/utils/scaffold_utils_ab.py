import os
import random
from copy import deepcopy
from pprint import pprint

import esm
import esm.inverse_folding
import torch

from byprot import utils
from byprot.datamodules.dataset.data_utils import PDBDataProcessor
from dataclasses import dataclass

# Import the DPLM-2 tokenizer
#from dplm2 import StructureTokenizer  # Assuming this is the correct import path
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2

@dataclass
class AntibodyMutation:
    single_res: list[str]
    cdr_regions: dict
    pdb_clean_path_str: str

    def __post_init__(self):
        self.chain_dict = {pdb: chain for pdb, chain in self.cdr_regions.items()}

    def get_intervals(self, list, single_res_domain=False):
        """Given a list (Tensor) of non-masked residues get new start and end index for motif placed in scaffold."""
        if single_res_domain:
            start = [l.item() for l in list]
            stop = start
        else:
            start = []
            stop = []
            for i, item in enumerate(list):
                if i == 0:
                    start.append(item.item())
                elif i == (len(list) - 1):
                    stop.append(item.item())
                elif i != len(list) and (item + 1) != list[i + 1]:
                    stop.append(item.item())
                    start.append(list[i + 1].item())
        return start, stop

    def mask_cdr_regions(self, pdb, sequence):
        """Mask the CDR regions in the antibody sequence."""
        start_idxs = self.cdr_regions[pdb]['start']
        end_idxs = self.cdr_regions[pdb]['end']
        sequence = list(sequence)

        end_idxs = [i + 1 for i in end_idxs]  # inclusive of final residue
        masked_sequence = sequence[:]
        for start, end in zip(start_idxs, end_idxs):
            masked_sequence[start:end] = ['<mask>'] * (end - start)
        
        return masked_sequence

    def prepare_data(self, pdb_path, alphabet, collator, num_seqs, device):
        structure = PDBDataProcessor().parse_PDB(pdb_path)
        batch = collator([deepcopy(structure) for _ in range(num_seqs)])
        prev_tokens, prev_token_mask = self._full_mask(batch["tokens"], batch["coord_mask"], alphabet)
        batch["prev_tokens"] = prev_tokens
        batch["prev_token_mask"] = prev_tokens.eq(alphabet.mask_idx)
        batch = utils.recursive_to(batch, device=device)
        return batch, structure["seq"]

    def _full_mask(self, target_tokens, coord_mask, alphabet):
        target_mask = (
            target_tokens.ne(alphabet.padding_idx)  # & mask
            & target_tokens.ne(alphabet.cls_idx)
            & target_tokens.ne(alphabet.eos_idx)
        )
        _tokens = target_tokens.masked_fill(target_mask, alphabet.mask_idx)
        _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
        return _tokens, _mask

    def get_initial_dplm2(self, args, aa_seq, struct_seq, tokenizer, pdb, device):
        init_aa_seq, init_struct_seq, scaffold_length_list = self.create_init_seq(pdb, aa_seq, struct_seq, tokenizer, args)
        batch = self.collate(tokenizer, init_aa_seq, init_struct_seq, args, device)
        pprint(batch)

        start_idxs_list, end_idxs_list = self.create_idxs_list(pdb, tokenizer, batch, args)
        batches = self.create_batches(batch, args)

        return batches, start_idxs_list, end_idxs_list, scaffold_length_list

    def create_init_seq(self, pdb, aa_seq, struct_seq, tokenizer, args):
        num = args.num_seqs
        aa_mask_token = tokenizer.aa_mask_token
        struct_mask_token = tokenizer.struct_mask_token

        init_aa_seq = []
        init_struct_seq = []
        scaffold_length_list = []
        for _ in range(num):
            scaffold_left_length = random.randint(5, 20)
            motif_aa_seq = self.mask_cdr_regions(pdb, aa_seq)
            motif_struct_seq = self.mask_cdr_regions(pdb, struct_seq)
            motif_overall_length = len(motif_aa_seq)

            scaffold_right_length = random.randint(5, 20)
            overall_length = scaffold_left_length + motif_overall_length + scaffold_right_length
            scaffold_length_list.append(scaffold_left_length + scaffold_right_length)

            seq = (
                [aa_mask_token] * scaffold_left_length
                + motif_aa_seq
                + [aa_mask_token] * scaffold_right_length
            )
            init_aa_seq.append("".join(seq))

            seq = (
                [struct_mask_token] * scaffold_left_length
                + motif_struct_seq
                + [struct_mask_token] * scaffold_right_length
            )
            init_struct_seq.append("".join(seq))

        return init_aa_seq, init_struct_seq, scaffold_length_list

    def collate(self, tokenizer, init_aa_seq, init_struct_seq, args, device):
        batch_aa = tokenizer.batch_encode_plus(
            init_aa_seq, add_special_tokens=False, padding="True", truncation="True", return_tensors="pt"
        )
        batch_aa = {
            "aa_ids": batch_aa["input_ids"],
            "aa_mask": batch_aa["attention_mask"].bool(),
            "aa_targets": batch_aa["input_ids"].clone(),
        }

        batch_struct = tokenizer.batch_encode_plus(
            init_struct_seq,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        batch_struct = {
            "struct_ids": batch_struct["input_ids"],
            "struct_mask": batch_struct["attention_mask"].bool(),
            "struct_targets": batch_struct["input_ids"].clone(),
        }
        batch = {
            "input_ids": torch.cat(
                (batch_struct["struct_ids"], batch_aa["aa_ids"]), dim=-1
            ),
            "input_mask": torch.cat(
                (batch_struct["struct_mask"], batch_aa["aa_mask"]), dim=-1
            ),
            "targets": torch.cat(
                (batch_struct["struct_targets"], batch_aa["aa_targets"]), dim=-1
            ),
        }
        batch.update(batch_struct)
        batch.update(batch_aa)

        batch["type_ids"] = ((batch["input_ids"] < 33) & batch["input_mask"]).int()
        batch["type_ids"].masked_fill_(~batch["input_mask"], 2)
        batch = utils.recursive_to(batch, device)

        aa_mask_idx = tokenizer.added_tokens_encoder[tokenizer.aa_mask_token]
        struct_mask_idx = tokenizer.added_tokens_encoder[tokenizer.struct_mask_token]
        partial_mask = (
            batch["input_ids"].ne(aa_mask_idx)
            & batch["input_ids"].ne(struct_mask_idx)
            & batch["input_ids"].ne(tokenizer.pad_token_id)
        ).type_as(batch["input_mask"])

        batch["partial_mask"] = partial_mask

        return batch

    def create_idxs_list(self, pdb, tokenizer, batch, args):
        single_res_domain = pdb in self.single_res

        start_idxs_list = []
        end_idxs_list = []
        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.added_tokens_encoder[tokenizer.aa_mask_token]
        bos_id = tokenizer.added_tokens_encoder[tokenizer.aa_cls_token]
        eos_id = tokenizer.added_tokens_encoder[tokenizer.aa_eos_token]
        get_intervals_seqs = batch["aa_ids"]

        for seq in get_intervals_seqs:
            nonmask_locations = (
                (seq != mask_id)
                & (seq != bos_id)
                & (seq != eos_id)
                & (seq != pad_id)
            ).nonzero().flatten() - 1
            new_start_idxs, new_end_idxs = self.get_intervals(
                nonmask_locations, single_res_domain=single_res_domain
            )
            start_idxs_list.append(new_start_idxs)
            end_idxs_list.append(new_end_idxs)

        return start_idxs_list, end_idxs_list

    def create_batches(self, batch, args):
        num = args.num_seqs
        batches = []
        start = 0
        end = start + args.batch_size
        while end < num:
            new_batch = {k: v[start:end] for k, v in batch.items()}
            batches.append(new_batch)
            start += args.batch_size
            end += args.batch_size
        if start < num:
            last_batch = {k: v[start:end] for k, v in batch.items()}
            batches.append(last_batch)
        return batches

def main():
    import test_struct_tokenizer
    # Example setup
    single_res = ["1qjg"]
    cdr_regions = {
        "1qjg": {"start": [37, 13, 98], "end": [37, 13, 98]},
        # Add more PDB entries with their CDR start and end indices
    }
    pdb_clean_path_str = "example.pdb"
    
    # Initialize AntibodyMutation
    antibody_mutation = AntibodyMutation(single_res, cdr_regions, pdb_clean_path_str)
    
    # Example arguments setup
    class Args:
        num_seqs = 10
        batch_size = 2
    
    args = Args()
    
    # Initialize the actual tokenizer from DPLM-2
    dplm2 = DPLM2.from_pretrained("airkingbd/dplm2_650m").cuda()

    tokenizer = dplm2.tokenizer
    struct_tokenizer = dplm2.struct_tokenizer

    #tokenizer = StructureTokenizer()  # Ensure this is the correct initialization
    aa_seq, struct_seq = test_struct_tokenizer.get_sequence_and_structure_tokens(pdb_clean_path_str)
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example PDB and sequences
    pdb = "1qjg"
    #aa_seq = "ACDEFGHIKLMNPQRSTVWY"
    #struct_seq = "HHHHHHHHHHHHHHHHHHHH"
    
    # Run the initial DPLM-2 process
    batches, start_idxs_list, end_idxs_list, scaffold_length_list = antibody_mutation.get_initial_dplm2(args, aa_seq, struct_seq, tokenizer, pdb, device)
    
    # Output results
    pprint(batches)
    pprint(start_idxs_list)
    pprint(end_idxs_list)
    pprint(scaffold_length_list)

if __name__ == "__main__":
    main()
