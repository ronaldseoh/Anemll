#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import os
import sys
import glob
import subprocess
from typing import List, Optional
import argparse

def compile_model(model_path: str, target_dir: str = "./") -> bool:
    """Compile a single CoreML model using coremlcompiler.
    
    Args:
        model_path: Path to .mlpackage file
        target_dir: Target directory for compiled model
    
    Returns:
        bool: True if compilation succeeded
    """
    try:
        cmd = ["xcrun", "coremlcompiler", "compile", model_path, target_dir]
        print(f"\nCompiling {os.path.basename(model_path)}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Compilation successful")
            return True
        else:
            print(f"Compilation failed with error:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error compiling {model_path}: {str(e)}")
        return False

def get_part_name(part: str) -> str:
    """Map part number to model name component.
    
    Args:
        part: Part number ("1", "2", "3")
    
    Returns:
        str: Model name component
    """
    part_map = {
        "1": "embeddings",
        "2": "FFN_PF",  # For combined FFN/prefill
        "3": "lm_head"
    }
    return part_map.get(part, part)

def find_chunk_models(lut_bits: int, num_chunks: int, part: str = "2", prefix: str = "llama") -> List[str]:
    """Find chunked model files matching pattern.
    
    Args:
        lut_bits: LUT quantization bits
        num_chunks: Number of chunks
        part: Model part (1=embeddings, 2=FFN/prefill, 3=lm_head)
        prefix: Model name prefix
    
    Returns:
        List of matching model paths
    """
    # For part 2, look for combined FFN_PF files
    if part == "2":
        pattern = f"{prefix}_FFN_PF_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
        models = sorted(glob.glob(pattern))
        if not models:
            print("No combined FFN_PF models found, looking for individual FFN and prefill models...")
            ffn_pattern = f"{prefix}_FFN_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
            prefill_pattern = f"{prefix}_prefill_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
            models = sorted(glob.glob(ffn_pattern) + glob.glob(prefill_pattern))
        return models
    else:
        part_name = get_part_name(part)
        pattern = f"{prefix}_{part_name}_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
        return sorted(glob.glob(pattern))

def compile_chunks(lut_bits: int, num_chunks: int, target_dir: str = "./") -> bool:
    """Compile all chunks of a model.
    
    Args:
        lut_bits: LUT quantization bits
        num_chunks: Number of chunks
        target_dir: Target directory for compiled models
    
    Returns:
        bool: True if all compilations succeeded
    """
    models = find_chunk_models(lut_bits, num_chunks)
    if not models:
        print(f"No chunk models found matching pattern")
        return False
        
    success = True
    for model in models:
        if not compile_model(model, target_dir):
            success = False
            
    return success

def compile_part(part: str, lut_bits: Optional[int] = None, target_dir: str = "./", prefix: str = "llama") -> bool:
    """Compile a specific model part.
    
    Args:
        part: Model part (1=embeddings, 2=FFN/prefill, 3=lm_head)
        lut_bits: Optional LUT quantization bits
        target_dir: Target directory for compiled model
        prefix: Model name prefix
    
    Returns:
        bool: True if compilation succeeded
    """
    # Get part name and construct model path
    part_name = get_part_name(part)
    lut_suffix = f"_lut{lut_bits}" if lut_bits else ""
    model_path = f"{prefix}_{part_name}{lut_suffix}.mlpackage"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
        
    return compile_model(model_path, target_dir)

def main():
    parser = argparse.ArgumentParser(description='Compile CoreML models to MLModelC format')
    parser.add_argument('part', type=str, help='Model part to compile (1=embeddings, 2=FFN, 3=lm_head)')
    parser.add_argument('--lut', type=int, help='LUT bits used in conversion (for part 2 and 3)')
    parser.add_argument('--chunk', type=int, help='Number of chunks (for part 2)')
    parser.add_argument('--prefix', type=str, default='llama', help='Prefix for model filenames')
    
    args = parser.parse_args()
    
    # Construct input filename based on part and parameters
    if args.part == '1':
        input_name = f'{args.prefix}_embeddings.mlpackage'
    elif args.part == '3':
        if args.lut is None:
            print("Error: --lut required for part 3")
            return 1
        input_name = f'{args.prefix}_lm_head_lut{args.lut}.mlpackage'
    elif args.part == '2':
        if args.lut is None or args.chunk is None:
            print("Error: --lut and --chunk required for part 2")
            return 1
        # For part 2, compile all chunks
        for i in range(args.chunk):
            chunk_name = f'{args.prefix}_FFN_PF_lut{args.lut}_chunk_{i+1:02d}of{args.chunk:02d}.mlpackage'
            compile_model(chunk_name)
        return 0
    else:
        print(f"Error: Invalid part {args.part}")
        return 1
    
    compile_model(input_name)
    return 0

if __name__ == "__main__":
    main() 