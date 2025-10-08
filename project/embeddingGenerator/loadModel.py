from sentence_transformers import SentenceTransformer

def loadModel(modelName, device='cpu'):
    model = SentenceTransformer(modelName, device=device, trust_remote_code=False)
    return model



# def _select_device(model_name: str) -> str:
#     """
#     Choose an execution device based on availability and model size.
#     - Force CPU for very large models to avoid GPU OOM unless explicitly requested.
#     """
#     huge_models = {
#         'Qwen3-Embedding-8B',
#         'E5-mistral-7b-instruct',
#     }

#     if PREFERRED_DEVICE.lower() == 'cpu':
#         return 'cpu'
#     if PREFERRED_DEVICE.lower() == 'cuda':
#         return 'cuda' if torch.cuda.is_available() else 'cpu'

#     # auto
#     if model_name in huge_models:
#         return 'cpu'
#     return 'cuda' if torch.cuda.is_available() else 'cpu'


# def _load_model_safely(model_id: str, device: str):
#     """
#     Try loading the model with mixed precision on CUDA; on failure, fall back to CPU.
#     """
#     # Prefer float16 on CUDA to reduce memory; CPU stays float32
#     model_kwargs = {}
#     if device == 'cuda':
#         model_kwargs['torch_dtype'] = torch.float16

#     try:
#         m = SentenceTransformer(model_id, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
#         return m, device
#     except RuntimeError as e:
#         # Typical CUDA OOM -> fallback to CPU
#         if 'CUDA out of memory' in str(e) or 'CUDA error' in str(e):
#             log.warning(f'CUDA issue while loading {model_id} on GPU; falling back to CPU. {e}')
#             try:
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#                 m = SentenceTransformer(model_id, device='cpu')
#                 return m, 'cpu'
#             except Exception as e2:
#                 raise e2
#         raise e