#!/bin/bash

# Garante que tudo está compilado e limpo
make clean > /dev/null
make all > /dev/null

OUTPUT="resultados_finais.csv"
echo "API,Tipo_Escalabilidade,Modelos,Blocos_Teams,Threads,Tempo_Segundos" > $OUTPUT

# Função para executar e logar
run_test() {
    API=$1
    TYPE=$2
    MODELS=$3
    BLOCKS=$4
    THREADS=$5
    
    echo "Rodando: $API $TYPE | Modelos: $MODELS | Blocos: $BLOCKS..."
    
    # Executa o programa
    OUTPUT_CMD=$(./neuralnetwork $API $MODELS $BLOCKS $THREADS)
    
    # Filtra o tempo da saída padrão
    TIME=$(echo "$OUTPUT_CMD" | grep "Tempo Total" | awk '{print $NF}' | sed 's/s//')
    
    # Salva no CSV
    echo "$API,$TYPE,$MODELS,$BLOCKS,$THREADS,$TIME" >> $OUTPUT
}

# ==========================================
# 1. ESCALABILIDADE FORTE (Strong Scaling)
# ==========================================
# Objetivo: Diminuir o tempo mantendo o problema fixo.
# Carga fixa alta para diluir o overhead inicial.
MODELS_STRONG=1000 

echo ">>> Iniciando Teste FORTE (Problema Fixo: $MODELS_STRONG modelos)..."

# CUDA Strong
for BLOCKS in 1 2 4 8 16 32 64 128; do
    run_test "cuda" "Forte" $MODELS_STRONG $BLOCKS 128
done

# OpenMP Strong
for TEAMS in 1 2 4 8 16 32 64 128; do
    run_test "openmp" "Forte" $MODELS_STRONG $TEAMS 128
done

# ==========================================
# 2. ESCALABILIDADE FRACA (Weak Scaling)
# ==========================================
# Objetivo: Manter o tempo constante aumentando o problema proporcionalmente aos recursos.
# Definimos uma carga fixa POR BLOCO.
# Baseado no seu teste anterior: 1000 modelos = 26s -> 100 modelos ~= 2.6s
PER_BLOCK=100

echo ">>> Iniciando Teste FRACO (Carga por Bloco: $PER_BLOCK modelos)..."

# CUDA Weak
for BLOCKS in 1 2 4 8 16 32 64 128; do
    # O problema total cresce junto com o número de blocos
    TOTAL_MODELS=$((BLOCKS * PER_BLOCK))
    run_test "cuda" "Fraca" $TOTAL_MODELS $BLOCKS 128
done

# OpenMP Weak
for TEAMS in 1 2 4 8 16 32 64 128; do
    TOTAL_MODELS=$((TEAMS * PER_BLOCK))
    run_test "openmp" "Fraca" $TOTAL_MODELS $TEAMS 128
done

echo "------------------------------------------------"
echo "Testes Finalizados! Confira o arquivo $OUTPUT"
echo "------------------------------------------------"
cat $OUTPUT