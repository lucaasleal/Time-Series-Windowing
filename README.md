# 📊 Accelerometer Time Series Windowing

## Dataset

Dataset used: **WISDM (Wireless Sensor Data Mining)**

🔗 Link: https://www.cis.fordham.edu/wisdm/dataset.php

---

## 📌 Objective

Prepare accelerometer data for Machine Learning models using:

* Normalization
* Sliding window segmentation
* Train/test split

---

## Data Organization

The original data is stored in `.txt` files with the following columns:

* `user` → user ID
* `activity` → performed activity
* `timestamp` → measurement time
* `x, y, z` → accelerometer values

---

## Data Shapes

### 🔹 Raw data

After loading and concatenation:

```
(N, 6)
```

Where:

* N = total number of samples
* 6 columns: user, activity, timestamp, x, y, z

---

### 🔹 After cleaning and selection

Keeping only:

```
[user, x, y, z]
```

Shape:

```
(N, 4)
```

---

### 🔹 Model input data

Removing the `user` column:

```
(N, 3)
```

Where:

* 3 features → x, y, z 

---

## Normalization

MinMax normalization (from Scikit-Learn) was applied in the range:

```
[-1, 1]
```

Using:

```python
MinMaxScaler(feature_range=(-1, 1))
```

---

## Window Segmentation

### Parameters:

* `window_size = 128`
* `step = 64` (50% overlap)

---

### Strategy:

* Data is separated by user
* Split:

  * 80% training
  * 20% testing
* Sliding windows are applied

---

## 📌 Shapes Throughout the Pipeline

### 🔹 Before windowing (per user)

```
(T, 3)
```

Where:

* T = number of samples for a given user
* 3 = x, y, z

---

### 🔹 Each window

```
(128, 3)
```

---

### 🔹 Final shapes

```
Shape de uma janela: (128, 3)

Shape Final do Treino: (59972, 128, 3)
Shape Final do Teste:  (14929, 128, 3)
```

---

## Code
Pipeline steps:

1. File loading
2. Data cleaning
3. Normalization
4. Window segmentation
5. Tensor creation

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/lucaasleal/Time-Series-Windowing.git
cd Time-Series-Windowing
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the script

```bash
python main.py
```

---

##  Notes

* Incomplete windows are discarded
* Overlapping windows increase dataset size
* The `user` column is only used for segmentation
* Each user is processed independently

---

## 🚀 Possible Extensions

* Train models (CNN, LSTM) (in future)
* Activity classification
* Signal visualization (x, y, z)
* Data augmentation (in future)

---

## Author
@lucaasleal

