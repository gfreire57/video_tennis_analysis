# Sistema de Reconhecimento de Golpes de Tênis

**Resumo executivo do projeto - Versão em Português**

---

## Visão Geral do Projeto

Este projeto implementa um sistema completo de reconhecimento automático de golpes de tênis usando **estimação de pose** (MediaPipe) e **redes neurais LSTM**. O sistema é capaz de identificar diferentes tipos de golpes (forehand, backhand, saque, slices) em vídeos contínuos de partidas de tênis.

### Objetivo Principal

Detectar e classificar automaticamente golpes de tênis em vídeos, gerando:
- Timeline visual dos golpes detectados
- Estatísticas de frequência de cada tipo de golpe
- Relatórios detalhados com timestamps
- Dados exportados em JSON para análises posteriores

---

## Como o Sistema Funciona

### Pipeline Completo

```
Vídeo de Tênis
    ↓
[1] Extração de Pose (MediaPipe)
    - Detecta 33 pontos do corpo (ombros, cotovelos, pulsos, quadris, etc.)
    - Gera 132 features por frame (x, y, z, visibilidade)
    ↓
[2] Criação de Sequências
    - Janelas deslizantes de 45 frames (~1.5 segundos)
    - Sobreposição de 50% para melhor detecção
    ↓
[3] Classificação LSTM
    - Rede neural aprende padrões temporais do movimento
    - Reconhece sequências específicas de cada golpe
    ↓
[4] Pós-processamento
    - Filtragem por confiança
    - Mesclagem de detecções próximas
    - Geração de timeline e relatórios
    ↓
Resultado: Golpes identificados com timestamps
```

---

## Arquitetura do Modelo

### Por Que LSTM?

Golpes de tênis são **sequências temporais** de movimentos corporais:
- **Preparação** (backswing): frames 0-15
- **Aceleração** (forward swing): frames 15-30
- **Contato**: frame ~25
- **Finalização** (follow-through): frames 30-45

**LSTM** (Long Short-Term Memory) é ideal porque:
- ✅ Lembra frames anteriores ao processar frames seguintes
- ✅ Aprende padrões de movimento ao longo do tempo
- ✅ Captura a sequência completa do golpe

### Arquitetura da Rede Neural

```
Entrada: (45 frames, 132 features) = Sequência de 1.5 segundos de movimento

    ↓
[LSTM Layer 1] - 64 unidades
    ↓ Dropout 0.4
[LSTM Layer 2] - 128 unidades
    ↓ Dropout 0.4
[LSTM Layer 3] - 64 unidades
    ↓ Dropout 0.3
[Dense Layer] - 64 unidades
    ↓ Dropout 0.3
[Dense Layer] - 64 unidades
    ↓ Dropout 0.2
[Output] - num_classes (softmax)

Saída: Probabilidades [P(forehand), P(backhand), P(saque), ...]
```

**Exemplo de saída**: `[0.08, 0.92]` → 92% forehand, 8% backhand

---

## Características Principais

### 1. Extração e Cache de Poses

**Problema**: Extrair poses de vídeos é muito lento (30-60 minutos para todos os vídeos).

**Solução**:
- Extrair poses **uma única vez** e salvar em disco (arquivos `.npz`)
- Carregar poses do disco para treinar (10-100x mais rápido!)

```bash
# Extrair poses (fazer uma vez)
poetry run python src/extract_poses.py

# Treinar (carrega do disco - rápido!)
poetry run python src/train_model.py
```

**Ganho**: De horas para minutos no treinamento!

### 2. Escalonamento por FPS

**Problema**: Vídeos com diferentes FPS (30, 48, 60 fps) têm velocidades diferentes.

**Solução**: Ajustar automaticamente o tamanho da janela baseado no FPS:
- Vídeo 30 fps: janela = 45 frames → 1.5 segundos
- Vídeo 60 fps: janela = 90 frames → 1.5 segundos (mesma duração!)

**Benefício**: Consistência temporal entre vídeos de diferentes câmeras.

### 3. Remoção da Classe "Neutro"

**Descoberta Crítica**:

❌ **Com classe "neutro"**:
```
Dados de treinamento:
  neutro: 9012 amostras (92%)  ← Esmagadora maioria
  forehand: 421 (4%)
  backhand: 347 (4%)

Resultado: Modelo aprende a sempre predizer "neutro" → 0% precisão nos golpes!
```

✅ **Sem classe "neutro"**:
```
Dados de treinamento:
  forehand: 420 (55%)  ← Balanceado
  backhand: 348 (45%)

Resultado: 83-88% de precisão nos golpes reais!
```

**Lição**: Treinar apenas nos golpes anotados, não nos frames vazios.

### 4. Rastreamento de Experimentos com MLflow

Todos os experimentos são rastreados automaticamente:

| Parâmetro | Exemplo |
|-----------|---------|
| window_size | 45 frames |
| learning_rate | 0.0005 |
| batch_size | 32 |
| fps_scaling | True |
| bidirectional | False |

| Métrica | Exemplo |
|---------|---------|
| test_accuracy | 85.4% |
| f1_score | 0.87 |
| training_time | 8.5 min |

**Comparar resultados**:
```bash
poetry run mlflow ui
# Abrir http://localhost:5000
```

### 5. Grid Search Automatizado

Testar múltiplas configurações automaticamente:

```bash
poetry run python src/grid_search.py --grid minimal
```

Testa combinações de:
- Tamanhos de janela (30, 45, 60 frames)
- Learning rates (0.001, 0.0005, 0.0001)
- Arquiteturas (camadas LSTM, Bidirectional)
- Batch sizes (16, 32, 64)

**Resultado**: Encontra automaticamente a melhor configuração.

---

## O Que o Modelo "Vê"

### Features Extraídas (132 por frame)

MediaPipe detecta **33 pontos do corpo**:

```
        Cabeça (0)
           •
           │
  (11)•───┼───•(12)  ← Ombros
      │   │   │
  (13)•   │   •(14)  ← Cotovelos
      │   │   │
  (15)•   │   •(16)  ← Pulsos
           │
  (23)•───┼───•(24)  ← Quadris
      │   │   │
  (25)•   │   •(26)  ← Joelhos
```

**Cada ponto tem 4 valores**:
- `x`: posição horizontal (0-1)
- `y`: posição vertical (0-1)
- `z`: profundidade relativa
- `visibility`: confiança da detecção (0-1)

**Total**: 33 pontos × 4 valores = **132 features por frame**

### Como o Modelo Distingue Golpes

**Forehand** (direita para frente):
- Pulso direito: x aumenta (0.3 → 0.7)
- Cotovelo direito: estende (z aumenta)
- Quadris: rotação no sentido horário
- Pé esquerdo: planta e empurra

**Backhand** (esquerda para frente):
- Pulso esquerdo: x diminui (0.7 → 0.3)
- Cotovelo esquerdo: estende
- Quadris: rotação anti-horária
- Pé direito: planta e empurra

**O modelo aprende automaticamente** esses padrões dos dados de treinamento!

---

## Fluxo de Trabalho

### 1. Preparação dos Dados

```bash
# 1. Anotar vídeos no Label Studio
label-studio start

# 2. Verificar anotações
poetry run python src/verify_annotation.py

# 3. Extrair poses (uma vez!)
poetry run python src/extract_poses.py
```

### 2. Treinamento

```bash
# Treinamento simples
poetry run python src/train_model.py

# Grid search (testar múltiplas configurações)
poetry run python src/grid_search.py --grid minimal
```

### 3. Detecção em Novos Vídeos

```bash
poetry run python src/detect_strokes.py video.mp4
```

**Saídas geradas**:
- `video_timeline.png` - Timeline visual colorido
- `video_report.txt` - Relatório com timestamps
- `video_strokes.json` - Dados em JSON

### 4. Análise de Resultados

```bash
# Abrir interface do MLflow
poetry run mlflow ui
```

Comparar:
- Acurácia entre diferentes configurações
- F1-score por classe
- Matrizes de confusão
- Tempo de treinamento

---

## Destaques Técnicos

### 1. Janelas Deslizantes com Sobreposição

```
Vídeo: [frame 0, frame 1, frame 2, ..., frame 300]

Janela 1: frames 0-44   (45 frames)
Janela 2: frames 23-67  (overlap de 22 frames)
Janela 3: frames 46-90
...
```

**Por quê 50% de sobreposição?**
- Captura golpes que começam no meio de uma janela
- Gera mais amostras de treinamento
- Detecção mais precisa durante inferência

### 2. Votação por Maioria para Labels

```
Janela: frames 30-74 (45 frames)
Labels: [F, F, F, F, ..., B, B, B]
        │   └─ 30 frames forehand
        └─ 15 frames backhand

Contagem:
  forehand: 30 (67%)
  backhand: 15 (33%)

Resultado: Label = FOREHAND (maioria > 50%)
```

### 3. Pós-processamento Inteligente

**Problema**: Janelas deslizantes geram múltiplas detecções do mesmo golpe.

**Solução**: Mesclar detecções próximas
```
Detecções brutas (antes):
  Forehand: frames 30-74  (conf: 88%)
  Forehand: frames 35-79  (conf: 92%)
  Forehand: frames 40-84  (conf: 85%)

Após mesclagem:
  Forehand: frames 30-84  (conf média: 88.3%)
```

### 4. Filtragem por Confiança e Duração

```python
CONFIG = {
    'confidence_threshold': 0.7,  # Mínimo 70% de confiança
    'min_stroke_duration': 13,    # Mínimo 13 frames
}
```

**Filtra**:
- Predições com baixa confiança (< 70%)
- Detecções muito curtas (< 13 frames = ~0.4 segundos)

---

## Otimizações Disponíveis

### 1. Balanceamento de Classes

```python
# Pesos automáticos para classes desbalanceadas
class_weights = compute_class_weight('balanced', classes, y_train)
```

**Benefício**: Classes minoritárias recebem mais peso no treinamento.

### 2. LSTM Bidirecional

```python
CONFIG = {
    'use_bidirectional': True,  # Processa sequência nas duas direções
}
```

**Ganho esperado**: +2-3% em F1-score
**Custo**: Treinamento ~2x mais lento

### 3. Ajuste de Hiperparâmetros

| Parâmetro | Valor Padrão | Quando Aumentar | Quando Diminuir |
|-----------|--------------|-----------------|-----------------|
| window_size | 45 | Golpes lentos | Golpes rápidos |
| learning_rate | 0.0005 | Treino instável | Convergência lenta |
| batch_size | 32 | Mais GPU/RAM | Menos memória |
| dropout | 0.4 | Overfitting | Underfitting |

### 4. Pré-processamento de Vídeos

Para vídeos escuros ou com jogador pequeno:

```bash
poetry run python src/preprocess_video.py input.mp4 output.mp4 \
    --auto-brighten \
    --static-zoom 1.5 \
    --fisheye
```

**Melhora**: Detecção de pose de 55% → 90%!

---

## Resultados Esperados

### Acurácia Típica

| Cenário | Acurácia | F1-Score |
|---------|----------|----------|
| Dataset balanceado | 83-88% | 0.85-0.88 |
| Com otimizações | 85-92% | 0.87-0.91 |
| Bidirectional LSTM | +2-3% | +0.02-0.03 |

### Tempo de Execução

| Tarefa | CPU | GPU |
|--------|-----|-----|
| Extração de poses (15 vídeos) | 60 min | 30 min |
| Treinamento (150 epochs) | 30 min | 8 min |
| Detecção (vídeo 5 min) | 3 min | 1 min |

### Exemplo de Saída

```
Relatório de Análise - Partida de Tênis
========================================

Vídeo: match_001.mp4
Duração: 5:23 (323 segundos)
Golpes detectados: 47

Timeline de Golpes:
  00:12.3 - 00:13.8 | Forehand (conf: 0.92)
  00:15.1 - 00:16.4 | Backhand (conf: 0.87)
  00:19.7 - 00:21.0 | Forehand (conf: 0.91)
  ...

Estatísticas:
  Forehand: 24 (51.1%)
  Backhand: 23 (48.9%)
```

---

## Por Que Esta Abordagem Funciona

### 1. Features Baseadas em Pose (não pixels brutos)

**Vantagens**:
- ✅ **47.000x menor** que vídeo bruto (132 vs 6.2M features)
- ✅ **Invariante à câmera** (funciona de qualquer ângulo)
- ✅ **Invariante à iluminação** (dia ou noite)
- ✅ **Foco no movimento** (ignora fundo, roupa, etc.)
- ✅ **Menos dados necessários** para treinar

### 2. LSTM para Padrões Temporais

```
Frame 0:  Pulso atrás    → LSTM lembra: "preparação"
Frame 15: Pulso no meio  → LSTM lembra: "aceleração"
Frame 30: Pulso à frente → LSTM conclui: "FOREHAND!"
```

**Outras arquiteturas**:
- CNN: Boa para imagens, não para sequências
- RNN simples: Esquece frames antigos (vanishing gradient)
- Transformer: Precisa de mais dados

**LSTM é ideal** para este caso de uso!

### 3. Sistema em Duas Etapas

**Etapa 1 - Treinamento**:
- Foco: "O que é este golpe?"
- Dados: Apenas segmentos anotados
- Objetivo: Classificação pura

**Etapa 2 - Inferência**:
- Foco: "Quando os golpes ocorrem?"
- Dados: Vídeo contínuo completo
- Objetivo: Detecção + classificação

**Benefício**: Cada etapa otimizada para seu objetivo específico.

---

## Lições Aprendidas

### ✅ O Que Funcionou

1. **Remover classe "neutro"** - Breakthrough de 54% → 85% acurácia
2. **Cache de poses** - 100x mais rápido experimentar
3. **Janelas com sobreposição** - Detecção mais robusta
4. **MLflow tracking** - Experimentos organizados e reproduzíveis
5. **Escalonamento por FPS** - Consistência entre vídeos

### ❌ O Que Não Funcionou

1. **Incluir classe "neutro"** - Desbalanceamento massivo (92% neutro)
2. **Processar todos os frames** - Desperdiça tempo em não-golpes
3. **TensorFlow GPU via Poetry** - Conflitos de dependências
4. **Janelas sem sobreposição** - Perde golpes nas bordas

### 💡 Decisões de Design

**Por que 45 frames?**
- Golpes de tênis duram 0.5-1.5 segundos
- 45 frames @ 30fps = 1.5 segundos
- Captura golpe completo (preparação → contato → finalização)

**Por que dropout 0.4?**
- Datasets pequenos tendem a overfit
- Dropout alto (0.4) força generalização
- Reduz overfitting sem prejudicar muito o treino

**Por que stride=5 na detecção?**
- Verifica a cada ~0.16 segundos
- Detecção precisa de início/fim do golpe
- Custo computacional aceitável

---

## Próximos Passos e Melhorias Futuras

### Curto Prazo

1. **Aumentação de dados** - Time warping, espelhamento
2. **Mecanismo de atenção** - Focar em frames importantes
3. **Ensemble de modelos** - Combinar múltiplos modelos

### Médio Prazo

4. **Avaliação de qualidade** - "Quão bom foi o golpe?"
5. **Rastreamento de múltiplos jogadores** - Duplas
6. **Integração com rastreamento de bola** - Melhor timing

### Longo Prazo

7. **Inferência em tempo real** - Streaming ao vivo
8. **Deploy mobile** - TensorFlow Lite para smartphones
9. **Transfer learning** - Pré-treino em datasets grandes

---

## Comandos Essenciais

```bash
# Instalação
poetry install

# Extração de poses (uma vez)
poetry run python src/extract_poses.py

# Treinar modelo
poetry run python src/train_model.py

# Detectar golpes
poetry run python src/detect_strokes.py video.mp4

# Grid search
poetry run python src/grid_search.py --grid minimal

# Interface MLflow
poetry run mlflow ui

# Pré-processar vídeos
poetry run python src/preprocess_video.py input.mp4 output.mp4 --auto-brighten
```

---

## Estrutura do Projeto

```
video_tennis_analysis/
├── src/
│   ├── train_model.py          # Treinamento principal
│   ├── detect_strokes.py       # Detecção em vídeos
│   ├── extract_poses.py        # Extração de poses (cache)
│   ├── grid_search.py          # Busca de hiperparâmetros
│   └── verify_annotation.py    # Verificar anotações
├── data/
│   └── videos/                 # Vídeos de treinamento (720p recomendado)
├── label_studio_exports/       # Anotações JSON do Label Studio
├── pose_data/                  # Poses extraídas (.npz)
├── output/                     # Modelo treinado + métricas
├── analysis_output/            # Resultados de detecção
├── mlruns/                     # Dados do MLflow
└── documentacao/               # Documentação completa (10 arquivos)
    ├── README_DOCS.md          # Índice de navegação
    ├── 00_GETTING_STARTED.md   # Início rápido
    └── ...                     # Guias detalhados
```

---

## Requisitos do Sistema

**Mínimos**:
- Python 3.11+
- 4GB RAM
- 10GB espaço em disco

**Recomendados**:
- GPU CUDA (10x mais rápido)
- 8GB+ RAM
- 20GB+ espaço em disco

**Dependências principais**:
- TensorFlow 2.17-2.18
- MediaPipe 0.10.21
- MLflow
- OpenCV

---

## Referências

### Documentação Completa

Veja [documentacao/README_DOCS.md](README_DOCS.md) para:
- Guia de início rápido
- Arquitetura detalhada
- Otimização de modelos
- Grid search avançado
- Rastreamento MLflow
- E mais 5 guias especializados

### Começar Agora

```bash
# 1. Instalar
poetry install

# 2. Ler o guia de início
cat documentacao/00_GETTING_STARTED.md

# 3. Treinar seu primeiro modelo
poetry run python src/train_model.py

# 4. Detectar golpes
poetry run python src/detect_strokes.py seu_video.mp4
```

---

**Desenvolvido para análise de desempenho em tênis usando Machine Learning** 🎾
