# 🎸 A Playlist Mais Alta
### Como usamos o YAMNet para medir a paixão dos fãs

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iqppruSoyMy4exTmd1ORqUSNzbcYJG-T?usp=sharing)

Material do workshop apresentado no **The Developers Life Weekend + GDG DevFest — Campo Mourão 2026**.

Neste workshop, você vai transformar áudio de shows ao vivo em dados usando o **YAMNet**, modelo de classificação de áudio do Google. O ponto de partida é um case real: a **Heineken® The Loudest Playlist**, no Rock in Rio 2026. Nesse projeto, a Plugh mediu a reação do público e usou os picos de engajamento para atualizar a playlist oficial do evento no Spotify.

---

## 🎯 O que você vai aprender

- Por que **volume (RMS/dBFS) não é engajamento** e o que a IA acrescenta
- Como carregar e usar o **YAMNet** com `tensorflow_hub`
- Como preparar o áudio no formato exigido pelo modelo: **16 kHz, mono, float32**
- Como agrupar as **521 classes** do YAMNet em métricas que fazem sentido para um show
- Como analisar áudio contínuo com **janelamento deslizante** e **suavização**
- Como transformar probabilidades em **eventos automáticos**, como "Coro da Plateia" e "Explosão / Transição"

---

## 🚀 Como começar

Não é preciso instalar nada. Todo o workshop roda no navegador.

1. Clique no botão **Abrir no Colab** acima.
2. Faça login com sua conta Google.
3. Vá em **Arquivo → Salvar uma cópia no Drive** para ter a sua versão editável.
4. Execute as células em ordem com `Shift + Enter`.

**Pré-requisitos:** uma conta Google e noções básicas de Python.

---

## 🛠️ Mão na massa

### 01 — Setup do ambiente
Configuração do Colab, das bibliotecas e do modelo.

```python
import tensorflow_hub as hub
import librosa

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
wav, sr = librosa.load(file, sr=16000, mono=True)  # ← obrigatório
```

> ⚠️ **Atenção:** `sr=16000` e `mono=True` são obrigatórios. Áudio em 44,1 kHz ou estéreo quebra o shape do tensor do YAMNet ou gera resultados incorretos.

### 02 — O experimento dos áudios estáticos
Classificação de três arquivos de teste (`Music.mp3`, `Applause.mp3` e `crowd.mp3`). O resultado sai em uma tabela com Pandas e em um gráfico de barras agrupadas com Matplotlib, que mostra a diferença entre a assinatura acústica do palco e a da plateia.

### 03 — A linha do tempo do show
Análise de um áudio longo, janela a janela. O script gera uma timeline com tempo, volume em dBFS, métricas de IA e eventos detectados, além de dois gráficos alinhados no tempo: um de reações do público e outro de volume.

---

## 🧠 Conceitos-chave

| Conceito | Valor | Por quê |
|---|---|---|
| Taxa de amostragem | 16 kHz, mono, float32 | Formato de entrada exigido pelo YAMNet |
| Janela de análise | 0,975 s (15.600 amostras) | Tamanho do bloco que o modelo "ouve" |
| Passo (hop) | 0,487 s (7.800 amostras) | 50% de overlap: nenhum evento cai no corte entre janelas |
| Suavização | Média móvel com `deque(maxlen=5)` (≈ 2,5 s) | Remove o efeito "serra elétrica" das oscilações |

### De 521 classes a 5 métricas

| Métrica | Classes do YAMNet |
|---|---|
| Palco / Performance | `Music`, `Guitar`, `Drum` |
| Coro da plateia | `Singing`, `Choir` |
| Euforia & gritos | `Cheering`, `Shout`, `Crowd` |
| Aprovação / Transições | `Applause`, `Clapping` |
| Intervalos / Pista | `Hubbub`, `Speech` |

---

## 🏗️ Do Colab para o show ao vivo

Em produção, o mesmo pipeline recebe áudio contínuo:
