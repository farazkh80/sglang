# pip install --no-deps --no-build-isolation "flashinfer_python @ git+https://github.com/cyx-6/flashinfer.git@48945059dfcf071c8878f1c0f2462ddea0155f81" --force-reinstall --no-cache-dir

pip install --no-deps --no-build-isolation "flashinfer_python==v0.2.14" --force-reinstall --no-cache-dir

# Verify installation
echo "Verifying TRTLLM MLA kernel availability..."
python -c "
import flashinfer
print(f'flashinfer version: {flashinfer.__version__}')
if hasattr(flashinfer.rope, 'mla_rope_quantize_fp8'):
    print('✅ TRTLLM mla_rope_quantize kernel is available!')
else:
    print('❌ TRTLLM mla_rope_quantize kernel not found')
    exit(1)
"

echo "✅ flashinfer updated successfully with TRTLLM MLA support!" 