# 🛡️ ARP Spoofing Detection Dashboard

A comprehensive Streamlit dashboard for detecting ARP spoofing attacks using machine learning models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
python run_dashboard.py
```
Or directly with Streamlit:
```bash
streamlit run streamlit_dashboard.py
```

### 3. Access the Dashboard
Open your browser and go to: `http://localhost:8501`

## 📊 Dashboard Features

### 1. **Data Overview** 📊
- Dataset statistics and sample counts
- Label distribution visualization
- Feature importance analysis
- Data quality metrics

### 2. **Model Training** 🤖
- Train multiple ML models:
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - Isolation Forest (Unsupervised)
  - Autoencoder (Unsupervised)
- Real-time performance metrics
- Model comparison table

### 3. **Visualizations** 📈
- Interactive confusion matrices
- ROC curves with AUC scores
- t-SNE data visualization
- Feature distribution plots

### 4. **Real-time Predictions** 🔮
- Input custom network traffic features
- Get predictions from all trained models
- Ensemble voting system
- Confidence scores

### 5. **Model Comparison** 📋
- Side-by-side performance metrics
- Interactive comparison charts
- Model recommendations
- Best model identification

## 🔧 Technical Details

### Features Used
- **Rate**: Packet rate per second
- **ARP**: ARP protocol flag (0/1)
- **IAT**: Inter-Arrival Time between packets

### Models Implemented
1. **Random Forest**: Ensemble of decision trees
2. **XGBoost**: Gradient boosting framework
3. **SVM**: Support Vector Machine with RBF kernel
4. **Isolation Forest**: Unsupervised anomaly detection
5. **Autoencoder**: Neural network for reconstruction error

### Data Preprocessing
- Heuristic-based labeling using quantile thresholds
- Standard scaling for feature normalization
- Handling of infinite and missing values
- PCA for autoencoder dimensionality reduction

## 📁 File Structure

```
arp-spoofing-detection/
├── streamlit_dashboard.py      # Main dashboard application
├── run_dashboard.py           # Dashboard launcher script
├── arp_spoofing_detection.py  # Original ML script
├── requirements.txt           # Python dependencies
├── ARP_Spoofing_train.pcap.csv # Training data
├── ARP_Spoofing_test.pcap.csv  # Test data
└── DASHBOARD_README.md        # This file
```

## 🎯 Usage Examples

### Training Models
1. Navigate to "🤖 Model Training" page
2. Click "Train All Models" button
3. View performance metrics table

### Making Predictions
1. Go to "🔮 Predictions" page
2. Input your network traffic features:
   - Rate: Packet rate (e.g., 100.0)
   - ARP: ARP protocol flag (0 or 1)
   - IAT: Inter-arrival time (e.g., 0.01)
3. View predictions from all models

### Analyzing Results
1. Check "📈 Visualizations" for confusion matrices and ROC curves
2. Use "📋 Model Comparison" to compare model performance
3. Review "📊 Data Overview" for dataset insights

## 🔍 Model Performance

The dashboard automatically calculates and displays:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🛠️ Customization

### Adding New Models
1. Add model training code in `train_models()` function
2. Include model in the models dictionary
3. Add prediction logic in the predictions section

### Modifying Features
1. Update the `features` list in `preprocess_data()`
2. Adjust the labeling heuristics in `label_row()`
3. Update the prediction input interface

### Styling
- Modify the CSS in the `st.markdown()` section
- Update colors and layout in Plotly figures
- Customize the page configuration

## 🐛 Troubleshooting

### Common Issues
1. **Data not loading**: Ensure CSV files are in the correct directory
2. **Models not training**: Check that all dependencies are installed
3. **Visualizations not showing**: Verify Plotly is installed correctly

### Performance Tips
- Use the cached functions for better performance
- Reduce t-SNE computation for large datasets
- Consider model selection for faster predictions

## 📈 Future Enhancements

- Real-time data streaming integration
- Model persistence and loading
- Advanced feature engineering
- Hyperparameter tuning interface
- Export functionality for results
- Alert system for detected attacks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
