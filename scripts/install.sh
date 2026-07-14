#!/bin/bash

# Script de instalación para YOLO Macroinvertebrados
# Autor: Kevin Galeano
# Proyecto: PINV01-1159

set -e  # Salir en caso de error

echo "🦐 Instalando YOLO Macroinvertebrados..."
echo "======================================"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar uv
print_status "Verificando uv..."
if ! command -v uv &> /dev/null; then
    print_error "uv no está instalado. Instálalo con: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

print_success "uv encontrado"

# Instalar dependencias (uv crea el entorno virtual .venv automáticamente)
print_status "Instalando dependencias..."
uv sync
print_success "Dependencias instaladas en .venv"

# Crear directorios necesarios
print_status "Creando directorios del proyecto..."
mkdir -p logs datasets models results
print_success "Directorios creados"

# Configurar archivo .env
print_status "Configurando variables de entorno..."
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        print_warning "Archivo .env creado desde env.example"
        print_warning "Por favor edita .env con tus credenciales de Roboflow"
    else
        print_warning "Archivo env.example no encontrado"
        print_warning "Por favor crea manualmente el archivo .env"
    fi
else
    print_success "Archivo .env ya existe"
fi

# Verificar CUDA (opcional)
print_status "Verificando CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
    print_success "CUDA $CUDA_VERSION detectado"
    print_success "GPU disponible para entrenamiento"
else
    print_warning "CUDA no detectado. El entrenamiento será más lento usando CPU"
fi

# Instalación completada
echo ""
echo "🎉 Instalación completada exitosamente!"
echo "======================================"
echo ""
echo "Para comenzar a usar el proyecto:"
echo "1. Configura tu API key de Roboflow en el archivo .env"
echo "2. Ejecuta el pipeline: uv run main.py --pipeline-complete"
echo ""
echo "Para más información, consulta el README.md"
echo ""
print_success "¡YOLO Macroinvertebrados está listo para usar!" 