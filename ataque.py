from modelo import fitness, carregar_modelo, mostra_imagem_alterada
from util import cria_individuo, seleciona_3_individuos, combinar, verificar_limites
import numpy as np

TAM_POPULACAO = 50
FATOR_MULTACAO = 0.8
CROSSOVER_RATE = 0.8
IMAGEM_IDX = 166 # qual imagem pegamos do data set do CIFAR-10
TOLERANCIA_MELHORIA = 0.001
LIMITE_EPOCAS_SEM_AVANCO = 7

# Até agora, pelo que vi o 165 é a imagem q ele acerta com mais dúvida 0.4
# 166 era 0.7 de certeza e com o pixel certo foi para 0.7 certeza em outra classe

melhor_fitness_global = float('-inf')
melhor_fitness_atual = float('-inf')
epocas_sem_melhoria = 0


carregar_modelo()
populacao = [cria_individuo() for _ in range(TAM_POPULACAO)]

for epoch in range(500):
    
    fitness_list = []

    for i in range( TAM_POPULACAO ):
        individuo = populacao[i]

        a, b, c = seleciona_3_individuos(individuo, populacao)
        mutante = a + FATOR_MULTACAO * (b - c)
        mutante = verificar_limites(mutante)

        trial = combinar(individuo, mutante, CROSSOVER_RATE)

        t_fitness = fitness(trial, IMAGEM_IDX)
        i_fitness = fitness(individuo, IMAGEM_IDX)

        if t_fitness > i_fitness:
            populacao[i] = trial
            fitness_list.append((t_fitness, trial))
        else:
            fitness_list.append((i_fitness, individuo))

    melhor_fitness_atual, melhor_atacante = max(fitness_list, key=lambda x: x[0])

    if epoch > 0:
        if melhor_fitness_atual > melhor_fitness_global + TOLERANCIA_MELHORIA:
            melhor_fitness_global = melhor_fitness_atual
            epocas_sem_melhoria = 0
        else:
            epocas_sem_melhoria += 1
        
    print("="*30)
    print(f"Época {epoch}")
    print(f"Melhor fitness: {melhor_fitness_atual:.4f}")
    print(f"Crosover rate: {CROSSOVER_RATE}")
    print(f"Épocas sem avanço: {epocas_sem_melhoria}")

    if epocas_sem_melhoria > 12 or melhor_fitness_atual > 0.6:
        print("*"*10)
        print(f"Melhor indivíduo da época: {melhor_atacante}")
        mostra_imagem_alterada(melhor_atacante, IMAGEM_IDX)
        print("*"*10)
        break

    if epocas_sem_melhoria > LIMITE_EPOCAS_SEM_AVANCO:
        CROSSOVER_RATE = 0.9
    else:
        CROSSOVER_RATE = 0.6


