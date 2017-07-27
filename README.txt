This is the source code for a thesis on preprint server, "RNN PFC models performing cognitive tasks showed common features in plastic changes, not in the constructed structures". URL

These scripts were mainly written in Python2.7 and C++ (only in rHebb model) and requires Theano, Numpy, Matplotlib, Pandas and Seaborn Python libraries. The analysis and visualization of the model structures in the paper were done additionally with Scipy, Statsmodels in Jupyter notebook.

We compared four simple RNN models (HF[1], pycog[2], pyrl[3] and rHebb[4]) in the paper. HF model was mounted with modifying scripts from theano-hf written by BL Nicolas on github (https://github.com/boulanni/theano-hf) [5].
1. Generate dataset for the context-dependent integration task with script "generate_dataset.py".
2. Train models with "train.py" by specifying path to datasets directory.
3. Inactivation experiment is performed with "inactivation.py" by specifying path to datasets, models and logs directories.

The other models were downloaded from github (pycog: https://github.com/xjwanglab/pycog;  pyrl: https://github.com/xjwanglab/pyrl; rHebb:  https://github.com/ThomasMiconi/BiologicallyPlausibleLearningRNN) and modified mainly for inactivation experiments. To used modified files in this study, please replace or add the same name files (to replace files) in the same name directories.

Github address: https://github.com/sakuroki/flexible_RNN

[1] Mante, V., Sussillo, D., Shenoy, K. V, & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 503(7474), 78–84.
[2] Song, H. F., Yang, G. R., Wang, X.-J. (2016). Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework. PLOS Computational Biology, 12(2), e1004792.
[3] Song, H. F., Yang, G. R., Wang, X.-J. (2017). Reward-based training of recurrent neural networks for cognitive and value-based tasks. eLife, 6, 679–684.
[4] Miconi, T. (2017). Biologically plausible learning in recurrent neural networks reproduces neural dynamics observed during cognitive tasks. eLife, 6, 229–256.
[5] Martens, J., & Sutskever, I. (2011). Learning Recurrent Neural Networks with Hessian-Free Optimization. In Proc. 28th Int. Conf. Machine Learn. (ICML, 2011).
