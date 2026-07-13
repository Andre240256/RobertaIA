# RobertaIA - Tutorial de Uso

Este guia mostra como preparar o ambiente, treinar os agentes, avaliar os modelos e inspecionar os resultados do projeto RobertaIA.

## Estrutura Principal

Os arquivos centrais do projeto estão em `src/`:

- `src/robertaEnv.py`: ambiente Gymnasium `RobertaEnv-v0`, com dinâmica física, estado, recompensa e renderização.
- `src/reward_shaping.py`: função de recompensa usada pelo ambiente.
- `src/edo_solver.py`: integrador RK4 usado na simulação da dinâmica.
- `src/render.py`: desenho da interface com Pygame.
- `src/stable3_aux_funcs.py`: utilitários de seed e criação de ambiente auxiliar.
- `src/model_SAC_functions.py`: configuração do agente SAC.
- `src/model_PPO_functions.py`: configuração do agente PPO.
- `src/training.py`: ponto de entrada para treino.
- `src/debuging.py`: ponto de entrada para avaliação e geração de gráficos.

## 1. Preparar o Ambiente

Como este projeto será enviado apenas com os códigos, o avaliador deve criar a virtualenv localmente e instalar as dependências a partir do `requirements.txt`:

```bash
cd /home/andreassis/RobertaIA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o sistema já tiver um ambiente Python ativo, basta garantir que ele esteja apontando para a versão correta e rodar apenas:

```bash
pip install -r requirements.txt
```

## 2. Treinar um Modelo

O treino é feito pelo `src/training.py`. O script registra automaticamente o ambiente `RobertaEnv-v0`, cria os ambientes vetorizados e salva o modelo final dentro de `logs/`.

### SAC

```bash
python src/training.py --algorithm SAC --timesteps 1000000 --lr 0.0003 --seed 42
```

### PPO

```bash
python src/training.py --algorithm PPO --timesteps 1000000 --lr 0.0003 --seed 42
```

### Parâmetros

- `--algorithm`: obrigatório, escolha entre `SAC` e `PPO`.
- `--timesteps`: obrigatório, quantidade total de passos de treino.
- `--lr`: opcional, taxa de aprendizado inicial. Padrão: `0.0003`.
- `--seed`: opcional, semente para reprodutibilidade. Padrão: `42`.

### Saída do treino

Cada execução cria uma pasta com nome parecido com:

```text
logs/SAC/lr0.0003_seed42_episodes1000000_12-07-2026-14:49:00/
```

Dentro dela ficam:

- `info.json`: parâmetros físicos e metadados do ambiente.
- `SAC_Roberta_lr0.0003_seed42.zip` ou `PPO_Roberta_lr0.0003_seed42.zip`: modelo final salvo.
- `best_model.zip`: melhor modelo encontrado durante a avaliação periódica.
- arquivos do TensorBoard dentro das subpastas de log.

## 3. Avaliar um Modelo Treinado

Depois do treino, use `src/debuging.py` para rodar uma demonstração do modelo e gerar gráficos da evolução do ângulo.

```bash
python src/debuging.py --model logs/SAC/lr0.0003_seed42_episodes1000000_12-07-2026-14:49:00/SAC_Roberta_lr0.0003_seed42.zip --episodes 10 --device cpu
```

### Parâmetros

- `--model`: obrigatório, caminho completo para o `.zip` do modelo.
- `--episodes`: opcional, quantidade de episódios de teste. Padrão: `10`.
- `--device`: opcional, `cpu` ou `cuda` para inferência. Padrão: `cpu`.

### O que o script faz

1. Abre a janela de renderização do Pygame.
2. Executa o modelo carregado no ambiente `RobertaEnv-v0`.
3. Registra o ângulo `phi` e o setpoint ao longo do episódio.
4. Salva os gráficos em uma pasta `images_test/` ao lado do modelo carregado.

## 4. TensorBoard

Durante o treino, o projeto grava métricas em `logs/`. Para visualizar os resultados:

```bash
tensorboard --logdir logs/
```

Depois abra `http://localhost:6006/` no navegador.

## 5. Observações Importantes

- Execute os comandos sempre na raiz do repositório, porque os imports usam caminhos relativos.
- O ambiente `RobertaEnv-v0` é registrado dentro dos scripts de treino e avaliação; não é preciso registrar manualmente.
- O treino está configurado para rodar em CPU nas funções de criação dos modelos.
- O arquivo `requirements.txt` contém as dependências necessárias para recriar o ambiente em outra máquina.
