# 📘 Next Word Prediction using LSTM (PyTorch)

This project implements a **Next Word Predictor** using an LSTM (Long Short-Term Memory) neural network in **PyTorch**. It takes a text corpus, builds a vocabulary, converts sentences into sequences of indices, trains an LSTM model, and predicts the next word for a given input sequence.

---

## 🚀 Features

* Tokenizes raw text using **NLTK**.
* Builds a vocabulary and converts tokens to indices.
* Prepares padded training sequences for model input.
* Implements a custom PyTorch **Dataset** and **DataLoader**.
* Defines an **LSTM-based model** with an embedding layer, LSTM, and a fully connected output layer.
* Trains the model using **CrossEntropyLoss** and **Adam optimizer**.
* Provides a prediction function to generate the next word given a sequence.
* Calculates model accuracy on the dataset.

---

## 📂 Project Structure

```
pytorch-lstm-next-word-predictor/
│
├── lstm_next_word_predictor.ipynb   # Main notebook
├── README.md                        # Documentation
```

---

## 🛠️ Requirements

* Python 3.8+
* PyTorch
* Numpy
* NLTK

Install dependencies:

```bash
pip install torch nltk numpy
```

---

## 📊 Dataset

The dataset is a text corpus (provided inside the code) describing a **Data Science Mentorship Program (DSMP 2023)**. You can replace it with any text of your choice for training.

---

## 🏗️ Model Architecture

* **Embedding Layer** → Converts word indices into dense vectors.
* **LSTM Layer** → Captures sequential dependencies.
* **Fully Connected Layer** → Predicts probability distribution over vocabulary.

---

## ▶️ Training

Run the training loop:

```python
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}")
```

---

## 🔮 Prediction

Use the model to predict the next word for a given sequence:

```python
prediction(model, vocab, "The course follows a monthly")
```

Example Output:

```
"The course follows a monthly subscription"
```

---

## 📈 Evaluation

Model accuracy is calculated using:

```python
accuracy = calculate_accuracy(model, dataloader, device)
print(f"Model Accuracy: {accuracy:.2f}%")
```

---

## 📌 Future Improvements

* Use a larger dataset for better accuracy.
* Implement **pretrained embeddings** (Word2Vec, GloVe).
* Extend to **character-level prediction**.
* Deploy as an **API** for interactive text prediction.

---

## 👨‍💻 Author

Developed by **Kumar Gaurav and inspired from campusX** ✨

