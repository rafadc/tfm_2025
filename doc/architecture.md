# Arquitectura

El codigo fuente estara en una carpeta ```src```.

Utilizaremos python 3.12
Utilizaremos UV para el control de dependencias y NO usaremos uv venv. Las dependencias de manejan con uv sync
El codigo se valida con ruff y siempre intentaremos que no haya warnings.
Se utilizara langchain para abstraer el proveedor para las llamadas al LLM. Se usara Ollama.
La configuracion se hara usando variables de entorno

