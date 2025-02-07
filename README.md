# Assessing the ability of language models to simulate animal adoption predispositions



## Summary

The article examines the capabilities of language models in the context of simulating animal adoption predispositions. Artificial intelligence, especially chatbots, play an important role in various industries, improving efficiency and automating decision-making processes. To evaluate their capabilities, an experiment was conducted in which language models answered 20 survey questions about animal adoption. The questions were randomly arranged and had unique identifiers. The results obtained from the models were compared to each other and contrasted with the results of the random selection. The study not only assessed the consistency and accuracy of the models' answers, but also highlighted potential algorithmic biases and the impact of how the questions were formulated on the results. It was discovered that some models showed clear preference biases, which could have important implications for their future use in decision-making processes. The source code alongside research results is available on the GitHub platform.

**Keywords:** *artificial intelligence, language models, AI bias, Web API, AI advisors, LMM, API, GPT-4o-Mini, Gemini-1.5-Flash, GPT, Gemini*

---

### Important:
Some of the files contain Polish lanugage text. To translate, you can use DeepL or OpenAI's Whisper model.

The "biased" folder contains output and results from 3 modes  using *questions_biased.json* qeustions set.
The "neutral" folder contains output and results from 3 modes  using *questions_neutral.json* qeustions set.
The "photos" folder contains example screenshots and created plots.

***Due to a slight mistake, questions 13. and 14. within their respective JSON files has been swapped. This slight inconvenience doesn't impact the data nor makes the study erroneous, it just requires extra work to swap the data for plots, tables and so on.***

---

## Build instructions:

**Recommended IDE:** IntelliJ IDEA 2022.2.1 (Community Edition) with Python Community Edition (222.3739.68) plugin.

**About project:**
* Console scripts.

**Dependencies / libraries:**
* openai (1.54.5)
* google-generativeai (0.8.3)
* matplotlib (3.9.3)
* seaborn (0.13.2)
* numpy (2.0.2)
* pandas (2.2.3)

**To use OpenAI Web API or Google Web API adoption survey modes in the *main.py* script you need to insert your keys into *openai_api_key* and *google_api_key* variables inside said script**

---