#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import coremltools as ct
import os
import sys
import pkg_resources

# Add package root to path when running as script
if __name__ == '__main__':
    import pathlib
    package_root = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.insert(0, package_root)
    from anemll.ane_converter.metadata import AddCombinedMetadata
else:
    from ..ane_converter.metadata import AddCombinedMetadata

def parse_model_args(args):
    """Parse command line arguments in the format name=model.mlpackage func=funcname."""
    models_dict = {}
    current_model = None
    
    for arg in args:
        if '=' not in arg:
            continue
            
        key, value = arg.split('=')
        if key.startswith('name'):
            current_model = value
            if current_model not in models_dict:
                models_dict[current_model] = {'path': value}
        elif key.startswith('func') and current_model is not None:
            models_dict[current_model]['function'] = value
            
    return models_dict

def combine_models_custom(models_dict):
    """Combine models based on provided dictionary of model paths and function names."""
    if not models_dict:
        print("Error: No valid model specifications provided!")
        return False
        
    desc = ct.utils.MultiFunctionDescriptor()
    models_found = False
    function_names = []
    
    # Add each model to the descriptor
    for model_info in models_dict.values():
        model_path = model_info.get('path')
        target_function_name = model_info.get('function')
        
        if not all([model_path, target_function_name]):
            print(f"Warning: Incomplete specification for model {model_path}, skipping...")
            continue
            
        if os.path.exists(model_path):
            models_found = True
            print(f"Adding model: {model_path} as function {target_function_name}")
            desc.add_function(
                model_path,
                src_function_name="main",
                target_function_name=target_function_name
            )
            function_names.append(target_function_name)
        else:
            print(f"Warning: Model {model_path} not found, skipping...")
    
    if not models_found:
        print("Error: No valid models found to combine!")
        return False
    
    # Set default function to the first specified function
    first_model = next(iter(models_dict.values()))
    desc.default_function_name = first_model.get('function')
    
    # Save the combined model
    output_path = "combined_model.mlpackage"
    print(f"\nSaving multifunction model to: {output_path}")
    combined_model = ct.utils.save_multifunction(desc, output_path)
    
    # Add metadata
    AddCombinedMetadata(combined_model, [model for model_info in models_dict.values()])
    combined_model.save(output_path)
    print("Done!")
    return True


def validate_chunk_files(num_chunks, lut_bits=None, mode=None, prefix='llama'):
    """Validate that all required files exist before starting combination.
    
    Args:
        num_chunks: Number of chunks to validate
        lut_bits: LUT quantization bits (optional)
        mode: Either 'FFN', 'prefill', or None for both
        prefix: Model name prefix (default: 'llama')
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    lut_suffix = f'_lut{lut_bits}' if lut_bits is not None else ''
    missing_files = []
    
    for chunk_idx in range(num_chunks):
        if mode in [None, 'FFN']:
            ffn_path = f'{prefix}_FFN{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage'
            if not os.path.exists(ffn_path):
                missing_files.append(ffn_path)
                
        if mode in [None, 'prefill']:
            prefill_path = f'{prefix}_prefill{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage'
            if not os.path.exists(prefill_path):
                missing_files.append(prefill_path)
    
    if missing_files:
        print("\nError: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
        
    return True

def combine_chunks(num_chunks, lut_bits=None, mode=None, prefix='llama'):
    """Combine model chunks."""
    try:
        # Process each chunk
        for chunk_idx in range(num_chunks):
            print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks}")
            
            try:
                # Get current working directory
                cwd = os.getcwd()
                
                # Construct model paths with full paths
                lut_suffix = f"_lut{lut_bits}" if lut_bits else ""
                ffn_path = os.path.join(cwd, f"{prefix}_FFN{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage")
                prefill_path = os.path.join(cwd, f"{prefix}_prefill{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage")
                
                # Use FFN_PF instead of 2 in output filename
                output_path = os.path.join(cwd, f"{prefix}_FFN_PF{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage")
                
                print("\nDEBUG: Loading source models...")
                # Load source models for metadata
                print(f"Reading metadata from FFN model: {ffn_path}")
                ffn_model = ct.models.MLModel(ffn_path, skip_model_load=True)  # Only load metadata
                print("DEBUG: FFN model loaded")
                
                print(f"Reading metadata from prefill model: {prefill_path}")
                prefill_model = ct.models.MLModel(prefill_path, skip_model_load=True)  # Only load metadata
                print("DEBUG: Prefill model loaded")
                
                print("\nDEBUG: Creating multifunction descriptor...")
                desc = ct.utils.MultiFunctionDescriptor()
                
                # Add models with their metadata
                print(f"Adding FFN model as 'infer' function")
                desc.add_function(ffn_path, "main", "infer")
                
                print(f"Adding prefill model as 'prefill' function")
                desc.add_function(prefill_path, "main", "prefill")
                
                # Set default function
                desc.default_function_name = "infer"
                
                # Save combined model
                tmp_path = "_tmp_combined_model.mlpackage"
                print(f"\nDEBUG: Saving combined model to: {tmp_path}")
                ct.utils.save_multifunction(desc,tmp_path)
                print("DEBUG: Initial save completed")
                
                # Load the combined model with specific function
                print("\nDEBUG: Loading model to add metadata...")
                combined_model = ct.models.MLModel(tmp_path, 
                                                 function_name="infer",  # Load prefill function
                                                 is_temp_package=True,    # Not a temporary package
                                                 compute_units=ct.ComputeUnit.CPU_AND_NE,  # Enable ANE
                                                 skip_model_load=True)    # Load full model
                print("DEBUG: Model loaded")
                
                # Add metadata and save
                print("\nDEBUG: Adding combined metadata...")
                AddCombinedMetadata(combined_model, [ffn_model, prefill_model])
                print("DEBUG: Metadata added")
                
                output_path = f"{prefix}_FFN_PF{lut_suffix}_chunk_{chunk_idx+1:02d}of{num_chunks:02d}.mlpackage"
                print(f"\nDEBUG: Saving final model to: {output_path}")
                combined_model.save(output_path)
                print("DEBUG: Final save completed")
                
                print(f"Successfully combined chunk {chunk_idx+1}")
                
            except Exception as e:
                print(f"\nError processing chunk {chunk_idx+1}: {str(e)}")
                import traceback
                print("\nFull traceback:")
                traceback.print_exc()
                return False
                
        return True
        
    except Exception as e:
        print(f"\nError during combination process: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if we're using the new model combination format
    if len(sys.argv) > 1 and '=' in sys.argv[1]:
        models_dict = parse_model_args(sys.argv[1:])
        if not combine_models_custom(models_dict):
            print("\nUsage examples:")
            print("1. Combine specific models:")
            print("   python combine_models.py name1=model1.mlpackage func1=func1 name2=model2.mlpackage func2=func2")
            print("\n2. Combine LUT models:")
            print("   python combine_models.py --chunk 2 --lut 6")
            sys.exit(1)
    
    # Parse command line arguments
    lut_bits = None
    num_chunks = None
    mode = None
    prefix = 'llama'  # default prefix
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--lut':
            lut_bits = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--chunk':
            num_chunks = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--mode':
            mode = sys.argv[i + 1]
            if mode not in ['FFN', 'prefill']:
                print("Error: mode must be 'FFN' or 'prefill'")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--prefix':
            prefix = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            print("\nUsage:")
            print("  python combine_models.py --chunk N [--lut N] [--mode FFN|prefill] [--prefix name]")
            print("\nOptions:")
            print("  --chunk N    Number of chunks to combine")
            print("  --lut N      LUT quantization bits (optional)")
            print("  --mode X     Mode: FFN or prefill (optional)")
            print("  --prefix X   Model name prefix (default: llama)")
            print("\nExamples:")
            print("  python combine_models.py --chunk 2 --lut 6")
            print("  python combine_models.py --chunk 2 --lut 6 --mode FFN")
            print("  python combine_models.py --chunk 2 --prefix llama32")
            sys.exit(1)
    
    if num_chunks is None:
        print("Error: --chunk argument is required")
        sys.exit(1)
        
    # Combine the chunks
    if combine_chunks(num_chunks, lut_bits, mode, prefix):
        print("\nAll chunks combined successfully!")
    else:
        print("\nCombination process failed.")
        sys.exit(1) 