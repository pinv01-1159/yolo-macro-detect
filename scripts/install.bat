@echo off
REM Script de instalación para YOLO Macroinvertebrados (Windows)
REM Autor: Kevin Galeano
REM Proyecto: PINV01-1159

echo 🦐 Instalando YOLO Macroinvertebrados...
echo ======================================

REM Verificar uv
echo [INFO] Verificando uv...
uv --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv no está instalado. Instálalo desde: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
echo [SUCCESS] uv encontrado

REM Instalar dependencias (uv crea el entorno virtual .venv automáticamente)
echo [INFO] Instalando dependencias...
uv sync
if errorlevel 1 (
    echo [ERROR] Error al instalar dependencias.
    pause
    exit /b 1
)
echo [SUCCESS] Dependencias instaladas en .venv

REM Crear directorios necesarios
echo [INFO] Creando directorios del proyecto...
if not exist logs mkdir logs
if not exist datasets mkdir datasets
if not exist models mkdir models
if not exist results mkdir results
echo [SUCCESS] Directorios creados

REM Configurar archivo .env
echo [INFO] Configurando variables de entorno...
if not exist .env (
    if exist env.example (
        copy env.example .env >nul
        echo [WARNING] Archivo .env creado desde env.example
        echo [WARNING] Por favor edita .env con tus credenciales de Roboflow
    ) else (
        echo [WARNING] Archivo env.example no encontrado
        echo [WARNING] Por favor crea manualmente el archivo .env
    )
) else (
    echo [SUCCESS] Archivo .env ya existe
)

REM Verificar CUDA (opcional)
echo [INFO] Verificando CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] CUDA no detectado. El entrenamiento será más lento usando CPU
) else (
    echo [SUCCESS] CUDA detectado
    echo [SUCCESS] GPU disponible para entrenamiento
)

REM Instalación completada
echo.
echo 🎉 Instalación completada exitosamente!
echo ======================================
echo.
echo Para comenzar a usar el proyecto:
echo 1. Configura tu API key de Roboflow en el archivo .env
echo 2. Ejecuta el pipeline: uv run main.py --pipeline-complete
echo.
echo Para más información, consulta el README.md
echo.
echo [SUCCESS] ¡YOLO Macroinvertebrados está listo para usar!
pause 