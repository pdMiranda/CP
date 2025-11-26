echo "Teams,Threads,Time" > results.csv

for teams in 2 4 8 16; do
for threads in 32 64 128 256; do
    echo "Testing teams=$teams threads=$threads"
    make cpp_gpu_program NUM_TEAMS=$teams NUM_THREADS=$threads
    
    # Executa o programa e captura o tempo
    { time ./cpp_gpu_program ; } 2> temp_time.txt
    
    # Extrai o tempo real da execução
    real_time=$(grep "real" temp_time.txt | awk '{print $2}')
    
    # Salva os resultados
    echo "$teams,$threads,$real_time" >> results.csv
    
    # Limpa arquivo temporário
    rm temp_time.txt
done
done
