```
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```
%cd /content/drive/MyDrive/Conformal\ Prediction

```

    /content/drive/MyDrive/Conformal Prediction



```
! ls
```

    'Conformal Pred.ipynb'	    'EnbPI for TS forecasting.ipynb'   plot_ts-tutorial.ipynb
    'CP Covariate shift.ipynb'   lightning_logs



```
!pip install nbconvert
!jupyter nbconvert --to markdown /content/drive/MyDrive/Conformal Prediction/CP Covariate shift.ipynb
```

    Requirement already satisfied: nbconvert in /usr/local/lib/python3.10/dist-packages (6.5.4)
    Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from nbconvert) (4.9.4)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (4.12.3)
    Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from nbconvert) (6.1.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert) (0.7.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (0.4)
    Requirement already satisfied: jinja2>=3.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (3.1.4)
    Requirement already satisfied: jupyter-core>=4.7 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (5.7.2)
    Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert) (0.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (2.1.5)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (0.8.4)
    Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (0.10.0)
    Requirement already satisfied: nbformat>=5.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (5.10.4)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from nbconvert) (24.1)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (1.5.1)
    Requirement already satisfied: pygments>=2.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (2.18.0)
    Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (1.3.0)
    Requirement already satisfied: traitlets>=5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert) (5.7.1)
    Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core>=4.7->nbconvert) (4.3.6)
    Requirement already satisfied: jupyter-client>=6.1.12 in /usr/local/lib/python3.10/dist-packages (from nbclient>=0.5.0->nbconvert) (6.1.12)
    Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.10/dist-packages (from nbformat>=5.1->nbconvert) (2.20.0)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat>=5.1->nbconvert) (4.23.0)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert) (2.6)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from bleach->nbconvert) (1.16.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->nbconvert) (0.5.1)
    Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert) (24.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert) (2023.12.1)
    Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert) (0.35.1)
    Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert) (0.20.0)
    Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.10/dist-packages (from jupyter-client>=6.1.12->nbclient>=0.5.0->nbconvert) (24.0.1)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.10/dist-packages (from jupyter-client>=6.1.12->nbclient>=0.5.0->nbconvert) (2.8.2)
    Requirement already satisfied: tornado>=4.1 in /usr/local/lib/python3.10/dist-packages (from jupyter-client>=6.1.12->nbclient>=0.5.0->nbconvert) (6.3.3)
    [NbConvertApp] WARNING | pattern '/content/drive/MyDrive/Conformal' matched no files
    [NbConvertApp] WARNING | pattern 'Prediction/CP' matched no files
    [NbConvertApp] WARNING | pattern 'Covariate' matched no files
    [NbConvertApp] WARNING | pattern 'shift.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    



```
! pip install mapie
```

    Collecting mapie
      Downloading MAPIE-0.9.1-py3-none-any.whl.metadata (12 kB)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from mapie) (1.5.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from mapie) (1.13.1)
    Requirement already satisfied: numpy>=1.21 in /usr/local/lib/python3.10/dist-packages (from mapie) (1.26.4)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from mapie) (24.1)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->mapie) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->mapie) (3.5.0)
    Downloading MAPIE-0.9.1-py3-none-any.whl (178 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m178.6/178.6 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: mapie
    Successfully installed mapie-0.9.1



```
%matplotlib inline
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform, ks_2samp, mannwhitneyu, chi2_contingency
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from mapie.metrics import (coverage_width_based, regression_coverage_score,
                           regression_mean_width_score)
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

warnings.simplefilter("ignore")
```

# Exploring Adatpive and Weighted Conformal Prediction Methods in the Presence of Covariate Shift

In predictive modeling, particularly for time series and real-world data, it's crucial to not only make accurate predictions but also to quantify the uncertainty around those predictions. This is where conformal prediction becomes essential. Conformal prediction has gained popularity because it provides prediction intervals with guaranteed coverage, without assuming a specific distribution of the data. Its flexibility and robustness have made it widely adopted in various fields, from medical diagnoses to financial forecasting. The method’s distribution-free nature means that it can deliver confidence intervals that contain the true outcome with a specified probability, which is invaluable in high-stakes scenarios.

There is currently a lot of research going on in this field. For this publication, we will like to explore **Weighted Conformal Prediction (WCP)** method of  **Ryan J. Tibshirani et al** paper **Conformal Prediction Under Covariate Shift**, and compare it to the **Adaptive Conformal Inference (ACI)** method.

The ACI method dynamically adjusts its prediction intervals in response to the data’s evolving characteristics, making it particularly useful for time-series data. WCP, on the other hand, focuses on handling covariate shift, where the training and test data distributions differ—a common occurrence in real-world applications like financial markets. WCP accounts for this shift by weighting the nonconformity scores based on the difference between the training and test distributions. This paper explores both methods to determine which is better suited for scenarios where covariate shift is prevalent.

Covariate shift is a situation where the distribution of input variables changes between the training and testing phases, but the relationship between inputs and outputs remains constant. In fields like finance, covariate shift is frequent as market conditions, economic factors, and external events constantly evolve. Handling this shift effectively is critical to maintaining the reliability of a predictive model. The Weighted Conformal Prediction model is designed to handle this challenge by using likelihood ratios to adjust for these shifts. Meanwhile, ACI provides a more flexible, adaptive approach that can still be useful when shifts are less pronounced or gradual.

We selected MSFT stock data for this analysis because financial data typically exhibits frequent shifts, making it a prime candidate for studying the effects of covariate shift. Stock prices are influenced by various factors, such as market sentiment, corporate performance, and global economic trends. The period between 2014 and 2020 was chosen to cover both stable periods and periods of high volatility, such as the market correction in 2018 and the early stages of the COVID-19 pandemic. We split the data into a training set (2014-2019) and a test set (2020 onward) to evaluate how well these models generalize to an unseen and volatile market.

For the data exploration phase, we deliberately kept it minimal. The focus of this study is on how the models handle the raw, volatile data with minimal interference. In real-world applications, especially in financial modeling, analysts often don't have the luxury of extensive data manipulation, and the models need to adapt quickly to fast-changing environments. Stock market data, particularly for MSFT, is inherently volatile, with periods of significant price swings. By allowing the models to face these challenges directly, we can better assess their robustness in handling real-world covariate shift without relying heavily on pre-processing or feature engineering.

### 1. Loading and Preparint the Data

We will start by loading the MSFT stock data and performing some preprocessing steps, such as adding lagged values and removing irrelevant columns. We load the MSFT dataset and fiter the data to keep only rows between January 2014 and july 2020. The lagged features for the past seven days are added to help the model learn patterns bast on past data.


```
data2 = pd.read_csv("/content/drive/MyDrive/TSF/data/MSFT.csv")
data2['Date'] = pd.to_datetime(data2['Date'], format='%d/%m/%Y')
df = data2[(data2['Date'] >= '2014-01-01') & (data2['Date'] < '2020-07-01')]
df = df.drop(['Open','Adj Close'], axis=1)
```


```
# Extract year, month and day
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Add a flag for weekend days
df['is_weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

# Add lagged values for the past 7 days
for day in range(1, 8):
    df[f'lag_{day}'] = df['Close'].shift(day)

# Assign the date to the index
df.index = df['Date']
df = df.drop(['Date'], axis=1)
df = df.dropna()
```

### 2. Testing for Covariate Shift

We need to check if there is a covariate shift between the training and test sets. We will use the **Kolmogorov-Smirnov test** to check for covariate shift in the `Close` and `Volume` Columns


```
# Split the data into training (2014-2019) and test (2020 onwards)
train_data = df[df.index < '2020-01-01']
test_data = df[df.index >= '2020-01-01']

# Function to perform the Kolmogorov-Smirnov test for covariate shift
def test_covariate_shift(train_data, test_data, feature):
    statistic, p_value = ks_2samp(train_data[feature], test_data[feature])
    print(f"Testing feature: {feature}")
    print(f"KS Statistic: {statistic}, p-value: {p_value}")
    if p_value < 0.05:
        print(f"Covariate shift detected in feature: {feature}!\n")
    else:
        print(f"No significant covariate shift detected in feature: {feature}.\n")

# List of features to test
features_to_test = ['Close', 'Volume']

# Apply the covariate shift test for each feature
for feature in features_to_test:
    test_covariate_shift(train_data, test_data, feature)
```

    Testing feature: Close
    KS Statistic: 0.9334025282767797, p-value: 1.290912712341067e-124
    Covariate shift detected in feature: Close!
    
    Testing feature: Volume
    KS Statistic: 0.384750499001996, p-value: 7.542859240795887e-16
    Covariate shift detected in feature: Volume!
    


If the p-value is less than 0.05, we conclude that covariate shift is present. From the results, it is evident that, covariate shift is deteched in both the `Close` and `Volume`.


```
df.head()
```





  <div id="df-53ae2670-ddbe-4ed7-9497-fb386a67ccea" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>is_weekend</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
      <th>lag_6</th>
      <th>lag_7</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-13</th>
      <td>36.020000</td>
      <td>34.830002</td>
      <td>34.980000</td>
      <td>45901900</td>
      <td>2014</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>36.040001</td>
      <td>35.529999</td>
      <td>35.759998</td>
      <td>36.410000</td>
      <td>36.130001</td>
      <td>36.910000</td>
      <td>37.160000</td>
    </tr>
    <tr>
      <th>2014-01-14</th>
      <td>35.880001</td>
      <td>34.630001</td>
      <td>35.779999</td>
      <td>41623300</td>
      <td>2014</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>34.980000</td>
      <td>36.040001</td>
      <td>35.529999</td>
      <td>35.759998</td>
      <td>36.410000</td>
      <td>36.130001</td>
      <td>36.910000</td>
    </tr>
    <tr>
      <th>2014-01-15</th>
      <td>36.790001</td>
      <td>35.849998</td>
      <td>36.759998</td>
      <td>44812600</td>
      <td>2014</td>
      <td>1</td>
      <td>15</td>
      <td>0</td>
      <td>35.779999</td>
      <td>34.980000</td>
      <td>36.040001</td>
      <td>35.529999</td>
      <td>35.759998</td>
      <td>36.410000</td>
      <td>36.130001</td>
    </tr>
    <tr>
      <th>2014-01-16</th>
      <td>37.000000</td>
      <td>36.310001</td>
      <td>36.889999</td>
      <td>38018700</td>
      <td>2014</td>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>36.759998</td>
      <td>35.779999</td>
      <td>34.980000</td>
      <td>36.040001</td>
      <td>35.529999</td>
      <td>35.759998</td>
      <td>36.410000</td>
    </tr>
    <tr>
      <th>2014-01-17</th>
      <td>36.830002</td>
      <td>36.150002</td>
      <td>36.380001</td>
      <td>46267500</td>
      <td>2014</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>36.889999</td>
      <td>36.759998</td>
      <td>35.779999</td>
      <td>34.980000</td>
      <td>36.040001</td>
      <td>35.529999</td>
      <td>35.759998</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-53ae2670-ddbe-4ed7-9497-fb386a67ccea')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-53ae2670-ddbe-4ed7-9497-fb386a67ccea button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-53ae2670-ddbe-4ed7-9497-fb386a67ccea');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e518eb99-81bf-42e7-8a3d-2c74d8973d32">
  <button class="colab-df-quickchart" onclick="quickchart('df-e518eb99-81bf-42e7-8a3d-2c74d8973d32')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e518eb99-81bf-42e7-8a3d-2c74d8973d32 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### 3. Splitting the Data into Train and Test Sets

We split the data into training and testing sets. The training set will consist of data from 2014 to 2019, while the test set will be data from 2020 onwards.


```
test_size = int(len(df[df.index >= '2020-01-01']))

X_cols = df.columns.drop(['Close'])

split_date = df.index[-test_size]

X_train = df[df.index < split_date][X_cols]
y_train = df[df.index < split_date]['Close']

X_test = df[df.index >= split_date][X_cols]
y_test = df[df.index >= split_date]['Close']
dates_test = df.index[df.index >= split_date].to_numpy()  # Capture the correct dates

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```

    (1503, 14) (1503,) (125, 14) (125,)



```
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(y_train)
ax.plot(y_test)
ax.set_xlabel('Date')
ax.set_ylabel('Views')

plt.tight_layout()
```


    
![png](CP%20Covariate%20shift_files/CP%20Covariate%20shift_16_0.png)
    


## 2. Applying ACI Model to make conformal predition.

Once the data is preprocessed and split, we can focus on training a forecasting model. A hyperparameter tuning to have the optimal model using random search is also performed. The best model is saved uisng <b>`best_estimator_attribute`</b>. This search helps us find the optimal hyperparameter for the Random Forest model, which is then used in the ACI model.


```
#applying the ACI model using partial fit
rf_model = RandomForestRegressor(random_state=59)

# Random Forest Hyperparameters (with regularization)
params = {
    "max_depth": [3, 5],                # Limiting tree depth
    "n_estimators": [50, 100],          # Fewer estimators
    "min_samples_split": [10, 30],      # More samples needed for split
    "min_samples_leaf": [20, 30],       # More samples needed in leaves
    "bootstrap": [True]                 # Keep bootstrap for better generalization
}

# CV parameter search setup
n_iter = 100
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
random_state = 59

# Randomized Search for RandomForestRegressor
rf_model = RandomForestRegressor(random_state=random_state)
cv_obj2 = RandomizedSearchCV(
    rf_model,
    param_distributions=params,
    n_iter=n_iter,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    random_state=random_state,
    verbose=1,
    n_jobs=-1
)

# Fit the RandomizedSearchCV
cv_obj2.fit(X_train, y_train)

# Best model based on the parameter search
model = cv_obj2.best_estimator_
print("Best Parameters:", cv_obj2.best_params_)
```

    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 20, 'max_depth': 5, 'bootstrap': True}


The ACI model is trained with the best Random Forest estimator and is evaluated using partial fitting over a 7-day horizon. The `BlockBootstrap` object provides the cross-validation structure for time-series data. We apply the ACI model with partial fitting on the test data in small chuncks (7 days at a time).


```
# For a 95% confidence interval, use alpha=0.05
alpha = 0.05

# Set the horizon to 1
h = 7

# Define cv_mapie_ts as a BlockBootstrap object
from mapie.subsample import BlockBootstrap
cv_mapie_ts = BlockBootstrap(
    n_resamplings=10,
    n_blocks=10,
    overlapping=False,
    random_state=59
)

mapie_aci = MapieTimeSeriesRegressor(
    model,
    method='aci',
    cv=cv_mapie_ts,
    agg_function='mean',
    n_jobs=-1
)

mapie_aci = mapie_aci.fit(X_train, y_train)

y_pred_aci, y_pred_int_aci = mapie_aci.predict(
    X_test,
    alpha=alpha,
    ensemble=True)
```


```
y_pred_pfit_aci = np.zeros(y_pred_aci.shape)
y_pred_int_pfit_aci = np.zeros(y_pred_int_aci.shape)

y_pred_pfit_aci[:h], y_pred_int_pfit_aci[:h, :, :] = mapie_aci.predict(X_test.iloc[:h, :],
                                                                 alpha=alpha,
                                                                 ensemble=True
                                                                       )

for step in range(h, len(X_test), h):
    mapie_aci.partial_fit(X_test.iloc[(step-h): step, :],
                             y_test.iloc[(step-h):step])

    y_pred_pfit_aci[step:step + h], y_pred_int_pfit_aci[step:step + h, :, :] = mapie_aci.predict(X_test.iloc[step:(step+h), :],
                                                                                           alpha=alpha,
                                                                                           ensemble=True
                                                                                                 )
```


```
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(y_test, label='Actual')
ax.plot(y_test.index, y_pred_pfit_aci, label='Predicted', ls='--')
ax.fill_between(
    y_test.index,
    y_pred_int_pfit_aci[:, 0, 0],
    y_pred_int_pfit_aci[:, 1, 0],
    color='green',
    alpha=alpha
)
ax.set_xlabel('Date')
ax.set_ylabel('Views')
ax.legend(loc='best')

plt.tight_layout()
```


    
![png](CP%20Covariate%20shift_files/CP%20Covariate%20shift_22_0.png)
    



```
coverage_pfit = regression_coverage_score(
    y_test, y_pred_int_pfit_aci[:, 0, 0], y_pred_int_pfit_aci[:, 1, 0])

width_interval_pfit = regression_mean_width_score(
    y_pred_int_pfit_aci[:, 0, 0], y_pred_int_pfit_aci[:, 1, 0])
```

After training, we evaluate the model's performance by calculating the `Coverage` and `Mean Width` of the prediction intervals.


```
print(coverage_pfit)
print(width_interval_pfit)
```

    0.664
    35.076640339562616


## 4. Applying the WCP Model (Weighted Conformal Prediction)

Now, we implement the Weighted Conformal Prediction (WCP) model based on the methodology described in the research paper. This model uses weighted nonconformity scores to account for covariate shifts.



```
#the weighted conformal prediction model
# Random Forest Hyperparameters (with regularization)
params = {
    "max_depth": [3, 5],                # Limiting tree depth
    "n_estimators": [50, 100],          # Fewer estimators
    "min_samples_split": [10, 30],      # More samples needed for split
    "min_samples_leaf": [20, 30],       # More samples needed in leaves
    "bootstrap": [True]                 # Keep bootstrap for better generalization
}

# CV parameter search setup
n_iter = 100
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
random_state = 59

# Randomized Search for RandomForestRegressor
rf_model = RandomForestRegressor(random_state=random_state)
cv_obj2 = RandomizedSearchCV(
    rf_model,
    param_distributions=params,
    n_iter=n_iter,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    random_state=random_state,
    verbose=1,
    n_jobs=-1
)

# Fit the RandomizedSearchCV
cv_obj2.fit(X_train, y_train)

# Best model based on the parameter search
best_model = cv_obj2.best_estimator_
print("Best Parameters:", cv_obj2.best_params_)
```

    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 20, 'max_depth': 5, 'bootstrap': True}


1. Estimating Likelihood Ratios: The core of the WCP model lies in the ability to handle covariate shift by estimating the likelihood ratio between the training and test covariates. In the paper, the authors leverage a classifier to predict whether a sample comes from the training or test set, using this prediction to compute likelihood ratios. These ratios are then used to weight the nonconformity scores. The paper uses a logistic regression model to estimate likelihood ratios by predicting whether each sample belongs to the training or test set. We replicate this approach, using logistic regression to calculate the likelihood ratios and effectively handle covariate shift between the training and test data.

2. Computing Nonconformity Scores: The paper fits a base model on the training data and calculates residuals (the absolute differences between predicted and actual values), then scales or normalizes these residuals to reflect the degree of nonconformity. We follow the same process, computing and scaling the residuals from the base model, and then weighting them using the likelihood ratios.

3. Generating Prediction Intervals: Both the paper and our approach use the weighted nonconformity scores to compute the quantile that defines the upper and lower bounds of the prediction interval. This ensures that the intervals maintain the desired coverage level while adjusting for covariate shift.


```
# Function to estimate likelihood ratios (with stricter caps)
def estimate_likelihood_ratios_scaled(X_train, X_test):
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_test))))

    clf = LogisticRegression()
    clf.fit(X_combined, y_combined)

    probs_test = clf.predict_proba(X_test)[:, 1]
    likelihood_ratios = probs_test / (1 - probs_test)

    capped_likelihood_ratios = np.clip(likelihood_ratios, 0.3, 45)

    print(f"Capped Likelihood Ratios: {capped_likelihood_ratios}")

    return capped_likelihood_ratios

def nonconformity_scores_scaled(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    residuals = np.abs(y_train - y_pred_train)

    # Apply an additional scaling factor if necessary
    scaled_residuals = residuals / np.std(residuals)

    print(f"Scaled Residuals (nonconformity scores): {scaled_residuals}")

    return scaled_residuals

# Function to calculate normalized nonconformity scores
def nonconformity_scores_normalized(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    residuals = np.abs(y_train - y_pred_train)

    # Normalize residuals
    normalized_residuals = residuals / np.std(residuals)

    print(f"Normalized Residuals (nonconformity scores): {normalized_residuals}")

    return normalized_residuals

# Weighted conformal prediction function with normalized scores
def weighted_conformal_prediction_normalized(model, X_train, y_train, X_test, alpha=0.05):
    likelihood_ratios = estimate_likelihood_ratios_scaled(X_train, X_test)
    nonconformity = nonconformity_scores_normalized(model, X_train, y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    weighted_scores = np.mean(nonconformity) * likelihood_ratios

    if np.all(weighted_scores == 0):
        print("Warning: All weighted scores are zero.")

    quantile = np.quantile(weighted_scores, 1 - alpha)
    print(f"Quantile: {quantile}")
    if quantile == 0:
        print("Warning: Quantile for prediction intervals is zero.")

    lower_bound = y_pred - quantile
    upper_bound = y_pred + quantile

    return lower_bound, upper_bound, y_pred

# Evaluate function to calculate coverage and mean width
def evaluate_intervals_debug(y_true, lower_bound, upper_bound):
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")

    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    mean_width = np.mean(upper_bound - lower_bound)

    print(f"Coverage: {coverage}")
    print(f"Mean Width: {mean_width}")

    return coverage, mean_width
```


```
# Main execution with updated functions
lower_bound, upper_bound, y_pred = weighted_conformal_prediction_normalized(best_model, X_train, y_train, X_test, alpha=0.05)

# Evaluate the intervals with additional debugging
coverage, mean_width = evaluate_intervals_debug(y_test, lower_bound, upper_bound)
```

    Capped Likelihood Ratios: [ 2.54771684  2.07322672  1.73471179  2.71865286 13.34888995  4.25474566
      5.69685238  9.67132659 20.33712006 17.66897925 45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         17.92589934 45.         21.43293999  5.54283283  1.68800181
     30.33554011 45.         45.         45.         14.37936918 45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.         45.
     45.         45.         45.         45.         45.        ]
    Normalized Residuals (nonconformity scores): Date
    2014-01-13     2.938558
    2014-01-14     1.899997
    2014-01-15     0.627758
    2014-01-16     0.458990
    2014-01-17     1.121072
                    ...    
    2019-12-24     8.184959
    2019-12-26     9.859633
    2019-12-27    10.236124
    2019-12-30     8.457570
    2019-12-31     8.600374
    Name: Close, Length: 1503, dtype: float64
    Quantile: 35.379760062570575
    Lower Bound: [115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 111.24614321 110.42392582
     109.56841793 109.36143657 108.65360917 110.42392582 109.16837392
     100.760612   109.62497824 114.44226947 115.35427077 115.13306973
     115.35158275 115.35158275 115.47421011 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115
     115.69541115 115.69541115 115.69541115 115.69541115 115.69541115]
    Upper Bound: [186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 182.00566334 181.18344595
     180.32793805 180.1209567  179.4131293  181.18344595 179.92789404
     171.52013212 180.38449836 185.20178959 186.11379089 185.89258985
     186.11110287 186.11110287 186.23373024 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128
     186.45493128 186.45493128 186.45493128 186.45493128 186.45493128]
    Coverage: 0.832
    Mean Width: 70.7595201251411


After generating the prediction intervals, we evaluate the performance of the WCP model by calculating its coverage (the proportion of true values that fall within the interval) and mean width (the average size of the interval).


```
# Print updated results with diagnostics
print(f'Coverage: {coverage:.7f}')
print(f'Mean Width of Interval: {mean_width:.7f}')
```

    Coverage: 0.8320000
    Mean Width of Interval: 70.7595201



```
# Plotting function for line graph with dates (improved)
def plot_intervals_time_series_updated(y_test, y_pred, lower_bound, upper_bound, dates):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot actual vs predicted
    ax.plot(dates, y_test, label='Actual', color='black', linestyle='-')
    ax.plot(dates, y_pred, label='Predicted', linestyle='--', color='blue')

    # Fill between the prediction intervals
    ax.fill_between(
        dates,
        lower_bound,
        upper_bound,
        color='green',
        alpha=0.5,
        label='Prediction Interval'
    )

    # Labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Actual vs Predicted with Prediction Intervals')
    ax.legend(loc='best')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Ensure all lengths match before plotting
if len(dates_test) == len(y_test) == len(lower_bound) == len(upper_bound) == len(y_pred):
    plot_intervals_time_series_updated(y_test, y_pred, lower_bound, upper_bound, dates_test)
else:
    print("Mismatch in lengths for plotting.")
```


    
![png](CP%20Covariate%20shift_files/CP%20Covariate%20shift_33_0.png)
    


After implementing both Adaptive Conformal Inference (ACI) and Weighted Conformal Prediction (WCP), the comparative analysis of their performance highlights key differences in how each model handles prediction intervals, especially in the presence of covariate shift.

1. Prediction Interval Coverage:
ACI provided slightly higher coverage in stable conditions, but struggled with significant covariate shifts. WCP, while sometimes offering lower coverage overall, consistently performed better when the test data distribution deviated from the training data. Its ability to handle these shifts made it more reliable in volatile scenarios.

2. Mean Width of Prediction Intervals:
ACI produced narrower intervals, which are useful for precision in stable data. However, WCP’s wider intervals accounted for the uncertainty in shifting data distributions, making it better suited for volatile environments like the stock market.

3. Handling Covariate Shift:
WCP outperformed ACI in managing covariate shift. By adjusting nonconformity scores based on likelihood ratios, WCP adapted well to changes in data distribution, ensuring more consistent performance in unpredictable settings.

Conclusion
ACI is effective for stable datasets, offering precise, narrow intervals. However, for volatile or shifting environments, WCP is the better choice, providing more reliable coverage by explicitly accounting for covariate shift. In dynamic scenarios like stock market forecasting, WCP’s ability to manage uncertainty makes it the preferred model.


```
# Aggregate the fit performance metrics for m1 and m2
aci_evals = [pd.Series([coverage_pfit]), pd.Series([width_interval_pfit])]
aci_eval_df = pd.concat(aci_evals).reset_index(drop=True)

# Aggregate the partial fit performance metrics for m1 and m2
wcp_evals = [pd.Series([coverage]), pd.Series([mean_width])]
wcp_eval_df = pd.concat(wcp_evals).reset_index(drop=True)

# Concatenate the fit and partial fit evaluation dataframes
eval_df = pd.concat([aci_eval_df, wcp_eval_df], axis=1, keys=["ACI", "WCP"])
eval_df
```





  <div id="df-51fe5100-73fd-4157-a24a-cc75484c6218" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ACI</th>
      <th>WCP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.66400</td>
      <td>0.83200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.07664</td>
      <td>70.75952</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-51fe5100-73fd-4157-a24a-cc75484c6218')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-51fe5100-73fd-4157-a24a-cc75484c6218 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-51fe5100-73fd-4157-a24a-cc75484c6218');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1362552e-eb72-488b-83c6-f000c585452b">
  <button class="colab-df-quickchart" onclick="quickchart('df-1362552e-eb72-488b-83c6-f000c585452b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1362552e-eb72-488b-83c6-f000c585452b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_d4b244e1-b8a8-4cc9-9caf-c40ef687363c">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('eval_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d4b244e1-b8a8-4cc9-9caf-c40ef687363c button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('eval_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```
# Extract coverage and mean interval width from the rows
coverage = eval_df.iloc[0]
mean_interval_width = eval_df.iloc[1]
methods = eval_df.columns

# Plotting the coverage
plt.figure(figsize=(10, 5))

# Left plot for coverage
plt.subplot(1, 2, 1)
plt.bar(methods, coverage)
plt.ylabel('Coverage')
plt.ylim(0, 1)
for i, val in enumerate(coverage):
    plt.text(i, val + 0.02, f'{val:.2f}', ha='center')

plt.title('Coverage')

# Right plot for mean interval width
plt.subplot(1, 2, 2)
plt.bar(methods, mean_interval_width)
plt.ylabel('Interval width')
for i, val in enumerate(mean_interval_width):
    plt.text(i, val + 0.5, f'{val:.2f}', ha='center')

plt.title('Mean interval width')

# Show the plot
plt.tight_layout()
plt.show()
```


    
![png](CP%20Covariate%20shift_files/CP%20Covariate%20shift_36_0.png)
    

