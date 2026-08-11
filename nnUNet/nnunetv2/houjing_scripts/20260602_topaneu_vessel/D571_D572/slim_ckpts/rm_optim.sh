
# Enable both recursive matching and empty result handling
shopt -s globstar nullglob

# Insdie nnUNet/, find all .pth files in data/results/ and its subdirectories
ckpt_files=( data/results/**/*.pth )

# Check if any files were actually found
if [ ${#ckpt_files[@]} -eq 0 ]; then
    echo "No checkpoint files found!"
    exit 1
fi

# Reset options back to default behavior
shopt -u globstar nullglob

# Print total number of files found
echo "Found ${#ckpt_files[@]} checkpoint files."

# Loop over the files safely
for ckpt_file in "${ckpt_files[@]}"; do
    echo "========================================"
    echo "= Processing checkpoint: $ckpt_file"
    echo "========================================"

    # Show original size in MB
    original_size=$(stat -c%s "$ckpt_file")
    echo "   Original size: $(echo "scale=2; $original_size / 1024 / 1024" | bc) MB"

    # modify checkpoint files in-place
    .venv/bin/python nnunetv2/houjing_scripts/modify_ckpt_attr.py \
        --ckpt_file "${ckpt_file}" \
        --new_ckpt_file "${ckpt_file}" \
        --func_name rm_network_keys_w_substr_and_optim
    
    # Show new size in MB
    new_size=$(stat -c%s "$ckpt_file")
    echo "   New size: $(echo "scale=2; $new_size / 1024 / 1024" | bc) MB"
done
