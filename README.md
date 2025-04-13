# Social Media Analytics Using Hawkes' Process

This project models the virality and popularity of tweeted YouTube videos using **self-exciting Hawkes Processes** and neural networks. It combines temporal point process theory with machine learning to predict view counts based on tweet activity and external engagement factors.

---

## Objective

To predict YouTube video popularity from Twitter interactions by:
- Modeling **retweet cascades** using Hawkes Processes
- Capturing external impact factors influencing popularity
- Enhancing prediction accuracy with **artificial neural networks (ANNs)**

---

## Methodology

### Dataset
- 130K+ tweets containing YouTube video links
- Features include: retweet timing, user metadata, tweet content, and external signals

### Modeling Steps
1. **Temporal Feature Engineering**  
   - Constructed 10+ features capturing user engagement and tweet bursts
   - Modeled self-excitation using **Hawkes Intensity Functions**

2. **Neural Network Prediction**  
   - Used ANN with **L-BFGS optimizer** to regress predicted view counts
   - Combined Hawkes-based features with contextual tweet metadata

3. **Evaluation & Refinement**  
   - Compared baseline models vs Hawkes-enhanced ANN
   - Reduced MSE loss by 25% with optimized temporal modeling

---

## Results

| Metric | Value |
|--------|-------|
| Average Prediction Error | **4.96%** |
| Median Prediction Error | **3.00%** |
| Loss Reduction (vs baseline) | **25%** |

The use of Hawkes Processes significantly improved the model's ability to capture virality dynamics and forecast popularity with minimal lag.

---

## Key Features

- Event modeling using **Hawkes point processes**
- **L-BFGS optimized ANN** for fast convergence
- External engagement feature extraction from tweet data
- Real-world application to **social media popularity forecasting**

---

## Tools & Technologies

- Python (NumPy, Pandas, Scikit-learn)
- Hawkes process implementation (tick or custom)
- Neural networks (PyTorch/TensorFlow)
- Data visualization (Matplotlib, Seaborn)

---

## Future Extensions

- Integrate natural language processing for tweet sentiment analysis
- Extend to other platforms (e.g., Reddit, Instagram)
- Compare with Poisson, Log-normal, and recurrent neural baselines
- Add a real-time forecasting pipeline using Twitter API

---

## 👤 Author

**Saksham Gupta**  
Final Year, B.S. in Mathematics and Computing  
Indian Institute of Technology, Kharagpur  
Email: gsaksham.iitkgp@gmail.com  
[LinkedIn](https://www.linkedin.com/in/saksham-gupta23) • [GitHub](https://github.com/saksham-gupta23)

---
