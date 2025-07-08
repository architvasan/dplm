import torch
import warnings
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn
from byprot.utils import recursive_to

def get_sequence_and_structure_tokens(pdb_path, aa_sequence=None):
    # 临时禁用回归权重检查
    import esm.pretrained
    original_has_regression = esm.pretrained._has_regression_weights
    esm.pretrained._has_regression_weights = lambda x: False
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dplm2 = DPLM2.from_pretrained("airkingbd/dplm2_650m").cuda()
        
        tokenizer = dplm2.tokenizer
        struct_tokenizer = dplm2.struct_tokenizer
        from byprot.utils.protein.evaluator_dplm2 import load_from_pdb
        feats = load_from_pdb(pdb_path, process_chain=struct_tokenizer.process_chain)
        batch_data = [feats]
        batch = collate_fn(batch_data)
        batch = recursive_to(batch, device=dplm2.device)

        struct_ids = struct_tokenizer.tokenize(
            batch["all_atom_positions"],
            batch["res_mask"],
            batch["seq_length"]
        )
        struct_seq = struct_tokenizer.struct_ids_to_seq(
            struct_ids.cpu().tolist()[0]
        )
          
        if aa_sequence is None:
            from byprot.datamodules.pdb_dataset import utils as du
            aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])
        else:
            aa_seq = aa_sequence
        
        return aa_seq, struct_seq
    
    finally:
        esm.pretrained._has_regression_weights = original_has_regression

if __name__ == "__main__":
    pdb_path = "example.pdb"
    sequence_tokens, structure_tokens = get_sequence_and_structure_tokens(pdb_path)
    
    print(sequence_tokens)
    print(structure_tokens)
