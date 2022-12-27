# Transformer for Astronomy
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

In this project, we performed the variable stars classification using an LSTM and DNN model on five sky-survey datasets (`ASAS`, `MACHO`, `WISE`, `GAIA` and `OGLE`). The model achieved `95%` validation accuracy on ASAS and `88%` accuracy on both the validation and test set of MACHO. We built a versatile data pipeline to perform data loading and transformation, taking into account the dataset imbalance, unequal lightcurve sequence length and the need for normalization, for both optimal training performance and predictive power. We also built a training scheme that is compatible with all data types supported by the data pipeline, provides logging of a large number of classification metrics and supports dynamic adjustment to the learning rate. The training scheme is also environment-aware, being able to take advantage of the computational power of TPUs or multiple GPUs when they are available. 

Below we list the summary reports and presentations developed throughout the project 
- [Report 1](__reference__/report_1.pdf): LSTM-DNN
- [Report 2](__reference__/report_2.pdf): Transformer Model
- [July_1st](__reference__/July_1st.pdf): LSTM-DNN on a dataset of sinusoidally varying time series
- [Aug_14th](__reference__/Aug_14th.pdf): Exploratory data analysis
- [Aug_21st](__reference__/Aug_21st.pdf): Introduction to the Transformer model I
- [Sep_4th](__reference__/Sep_4th.pdf): Introduction to the Transformer model II
- [Sep_18th](__reference__/Sep_18th.pdf): Experimental study with the Transformer model
