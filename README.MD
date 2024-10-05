# Identificação de Doenças em Plantas com Redes Neurais Convolucionais

![Banner do Projeto](dataset-cover.png)

## Índice

- [Introdução](#introdução)
- [Objetivo do Projeto](#objetivo-do-projeto)
- [Dataset](#dataset)
- [Como Funciona](#como-funciona)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Treinamento e Avaliação](#treinamento-e-avaliação)
- [Resultados](#resultados)
- [Conclusão](#conclusão)
- [Como Executar](#como-executar)
- [Contato](#contato)

---

## Introdução

Este projeto foi desenvolvido para identificar **doenças em plantas** a partir de imagens de folhas. Utilizando técnicas de **Visão Computacional** e **Inteligência Artificial**.

Este README foi desenvolvido com o intuito de ser um material de apresentação do projeto para recrutadores, mostrando a capacidade de resolver problemas reais usando IA, de forma clara.

---

## Objetivo do Projeto

O objetivo principal deste projeto é desenvolver um modelo que seja capaz de **classificar imagens de folhas** em diferentes categorias, como "saudável" ou "infectada por alguma doença".

Por meio deste projeto, pude aplicar e aprimorar minhas habilidades em **machine Learning**, **visão computacional**, *inteligência artificial*.

---

## Dataset

Utilizamos o **PlantVillage Dataset**, um dos maiores e mais utilizados conjuntos de dados para o estudo de doçenças em plantas. Esse dataset possui milhares de imagens de folhas, tanto **saudáveis** quanto **afetadas por diferentes doenças**.

- **Plantas Incluídas**: Tomate, Maçã, Uva, Pimenta, Batata, entre outras.
- **Classes**: Cada planta possui imagens em categorias como "saudável" ou com doenças específicas, como "mancha preta", "míldio", etc.

As imagens têm uma resolução de 256x256 pixels e estão divididas em várias classes, facilitando a criação de modelos de classificação.

[Dowload do DataSet](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
---

## Como Funciona

1. **Coleta de Dados**: O dataset é carregado e dividido entre conjuntos de **treinamento**, **validação**, e **teste**.
2. **Processamento das Imagens**: As imagens são processadas para padronização, incluindo redimensionamento e normalização dos pixels.
3. **Treinamento do Modelo**: Utilizamos uma **Rede Neural Convolucional (CNN)**, treinada para aprender padrões que diferenciam folhas saudáveis das doentes.
4. **Predição**: Depois de treinado, o modelo é capaz de receber novas imagens de folhas e classificar se estão saudáveis ou com alguma doença.

---

## Arquitetura do Modelo

O modelo desenvolvido é uma **Rede Neural Convolucional (CNN)**, escolhida por ser particularmente eficaz em tarefas de **classificação de imagens**. A CNN foi composta por:

- **Camadas Convolucionais**: Para extrair características das imagens.
- **Camadas de Pooling**: Para reduzir as dimensões das características, evitando sobreajuste.
- **Camada Flatten**: Para converter as características 2D em um vetor 1D.
- **Camadas Densas**: Para realizar a classificação.
- **Camada Dropout**: Para prevenir overfitting.

**[Ver Código do Modelo](src/model.py)**

---

## Treinamento e Avaliação

### Treinamento

Para o treinamento, usamos as imagens já existentes no dataset, aplicando técnicas de **data augmentation** para melhorar a capacidade do modelo de generalizar. O processo de treinamento foi gerenciado pelo script `train.py`.

**[Ver Código do Treinamento](src/train.py)**

### Avaliação

O modelo foi avaliado no conjunto de teste para medir sua precisão e capacidade de generalização. As principais métricas usadas foram:

- **Acurácia**: A porcentagem de previsões corretas.
- **Matriz de Confusão**: Para visualizar como o modelo performou em cada classe.

**[Ver Código de Avaliação](src/evaluate.py)**

---

## Resultados

### Acurácia

- **Acurácia no Treinamento**: ~92%
- **Acurácia no Conjunto de Teste**: ~88%

### Visualizações

- **Histórico de Treinamento**: A evolução da perda e da acurácia durante as épocas.

![Histórico de Treinamento](reports/training_history.png)

- **Matriz de Confusão**: Para avaliar a performance em cada classe.

![Matriz de Confusão](reports/confusion_matrix.png)

**[Ver Relatório Completo de Avaliação](reports/classification_report.txt)**

---

## Conclusão

O projeto foi capaz de atingir uma boa acurácia na identificação de doenças em plantas, mostrando a eficácia das Redes Neurais Convolucionais para aplicações na agricultura.

**Principais Conquistas:**

- Desenvolvimento de um modelo que ajuda na identificação de doenças, oferecendo suporte a agricultores.
- Implementação de boas práticas de treinamento e avaliação de redes neurais.
- Geração de relatórios e visualizações para facilitar a compreensão dos resultados.

Este projeto demonstra como a **Inteligência Artificial** pode ser utilizada para resolver problemas do mundo real, melhorando a eficiência e a produtividade na agricultura.

---

## Como Executar

1. **Clone o Repositório**:

   ```bash
   git clone https://github.com/seuusuario/plant-disease-identification.git
   cd plant-disease-identification
   ```

2. **Instale as Dependências**:

   Recomenda-se usar um ambiente virtual:

   ```bash
   python -m venv env
   source env/bin/activate  # No Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Treine o Modelo**:

   Execute o script de treinamento:

   ```bash
   python src/train.py
   ```

4. **Avalie o Modelo**:

   Execute o script de avaliação:

   ```bash
   python src/evaluate.py
   ```

---

## Contato

- **Email**: seu.email@exemplo.com
- **LinkedIn**: [Seu Perfil no LinkedIn](https://www.linkedin.com/in/seuperfil)

Se você tiver alguma dúvida ou sugestão sobre o projeto, sinta-se à vontade para entrar em contato. Este projeto está aberto a colaborações e novas ideias!

**[Veja no GitHub](https://github.com/seuusuario/plant-disease-identification)**

---

**Licença**:

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---
