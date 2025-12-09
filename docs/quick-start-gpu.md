# Quick Start GPU

Install GPU extras and launch the API with CUDA enabled.

```bash
pip install -r requirements.txt '.[gpu]'
uvicorn datacreek.api:app --reload
```
