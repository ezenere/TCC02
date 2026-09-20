# Demo de webcam — reconhecimento de gestos ao vivo

Aplicação mínima para testar ao vivo qualquer modelo produzido pelo projeto.

```bash
conda activate tcc
python app/webcam_demo.py                  # menu: escolha o modelo; só então a câmera abre
python app/webcam_demo.py --list           # lista os modelos disponíveis e sai
python app/webcam_demo.py --model runs/eixo1_resnet50_s0/checkpoints/best.pt
python app/webcam_demo.py --model runs/eixo1_densenet121_s0/model_int8_fbgemm.pt   # int8, CPU
python app/webcam_demo.py --model runs/eixo1_resnet50_s0/trt/model_int8.engine      # TensorRT, GPU
python app/webcam_demo.py --source 1       # outra câmera (índice de /dev/videoN)
```

O modelo é escolhido **antes** de iniciar; para testar outro, feche (`q`) e abra de novo.

| artefato | backend | onde roda |
|---|---|---|
| `runs/<run>/checkpoints/best.pt` | PyTorch (inclui os modelos podados) | GPU se houver, senão CPU (`--device cpu` força) |
| `runs/<run>/model_int8_fbgemm.pt` | TorchScript int8 | CPU |
| `runs/<run>/model.onnx` | ONNX Runtime | CPU |
| `runs/<run>/trt/model_{fp32,fp16,int8}.engine` | TensorRT | GPU |

**Como usar.** As redes foram treinadas em um recorte quadrado em volta da mão (bbox + 10% de margem), não na cena inteira. O app
classifica o que está dentro do **quadrado verde**: coloque a mão ali, ocupando a maior parte do quadrado. Teclas: `q`/`ESC` sai,
`+`/`-` muda o tamanho do quadrado, `m` liga/desliga o espelho, clique do mouse move o quadrado. A barra lateral mostra as 3 classes
mais prováveis; o rodapé mostra modelo, backend, latência por quadro e FPS. As probabilidades são suavizadas entre quadros (`--smooth`).

**Limitações honestas.** (1) Não há classe "sem gesto" entre as 18: com a mão fora do quadrado a rede ainda escolhe um gesto; abaixo de
`--threshold` (60%) o rótulo vira `?`. (2) Não há detector de mão — o quadrado é fixo. Um detector (ex.: MediaPipe Hands) não foi incluído
porque fixa `numpy<2` e quebraria o ambiente pinado do projeto. (3) O pré-processamento replica a avaliação: redimensiona o quadrado
para 256 e usa o centro 224×224.

**Sem câmera / teste automático.** `--source` aceita imagem, vídeo ou pasta de imagens, e `--headless` imprime uma linha por quadro:

```bash
python app/webcam_demo.py --model runs/eixo1_resnet50_s0/checkpoints/best.pt --source data/processed/test/ok --roi 1.0 --headless
```
Verificado assim com um crop de teste por classe: 18/18 corretos nos quatro backends (PyTorch, int8 TorchScript, ONNX Runtime, TensorRT int8).
