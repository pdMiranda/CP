# Neural Network C++ (GPU Edition: CUDA & OpenMP)

Este projeto é uma evolução da implementação de Rede Neural em C++ (Projeto 01), reestruturada para execução paralela massiva em GPUs. O objetivo principal é testar a **escalabilidade** treinando múltiplas instâncias de redes neurais (modelos) simultaneamente, comparando duas abordagens de paralelismo:

1.  **CUDA**: Utiliza kernels nativos da NVIDIA para gerenciar blocos e threads.
2.  **OpenMP**: Utiliza diretivas de compilação (`#pragma omp target`) para descarregar o processamento para a GPU.

## 📋 Requisitos

Para compilar e executar este projeto, você precisará de:

* **Sistema Operacional**: Linux ou Windows com WSL 2 (Windows Subsystem for Linux).
* **Compilador C++**: `g++` (com suporte a OpenMP).
* **CUDA Toolkit**: `nvcc` (Compilador da NVIDIA).
* **Drivers NVIDIA**: Instalados e configurados corretamente no sistema/WSL.

## ⚙️ Compilação

O projeto utiliza um `Makefile` híbrido que gerencia a compilação de arquivos `.cpp` (OpenMP/Host) e `.cu` (CUDA).

Para compilar o projeto:

```bash
make all