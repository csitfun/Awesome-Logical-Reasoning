# ✨Awesome-Logical-Reasoning  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Paper list for logical reasoning

-- [Survey](https://arxiv.org/abs/2502.09100)

**Key Words**: premise, conclusion, argument, reasoning, inference, proposition, forward chaining, backward chaining, critical thinking, syllogism

![](https://img.shields.io/github/last-commit/csitfun/Awesome-Logical-Reasoning) Maintained by [Hanmeng Liu](https://scholar.google.com/citations?user=vjmL_9UAAAAJ&hl=en), [Ruoxi Ning](https://ruoxining.github.io)

![](https://img.shields.io/badge/PRs-Welcome-red) Welcome to contribute!

![](https://github.com/csitfun/Awesome-Logical-Reasoning/blob/main/assets/Logical_Reasoning.png)
## ✨Contents
- [VERBAL REASONING](#verbal-reasoning)
  - [Surveys](#surveys)
  - [Formal Logical Reasoning (Deductive Reasoning)](#formal-logical-reasoning-(deductive-reasoning))
  - [Informal Logical Reasoning](#informal-logical-reasoning)
    - [Inductive Reasoning](#inductive-reasoning)
    - [Abductive Reasoning](#abductive-reasoning)
    - [Analogical Reasoning](#analogical-reasoning)
  - [Logical Fallacy](#logical-fallacy)
  - [Argument](#argument)
  - [Inference](#inference)
  - [Legal](#legal)
  - [Critical Thinking](#critical-thinking)
  - [Theorem Proving](#theorem-proving)
  - [Natural Language Inference](#natural-language-inference)
    - [Multi-Choice Question](#multi-choice-question)
    - [Converted from QA](#converted-from-qa)
    - [Converted from Summarization](#converted-from-summarization)
    - [Tabular Premises (Tabular Reasoning)](#tabular-premises-(tabular-reasoning))
    - [Fact-checking](#fact-checking)
    - [Probabilistic NLI](#probabilistic-nli)
    - [Document-level NLI](#document-level-nli)
    - [Complex Reasoning NLI](#complex-reasoning-nli)
    - [Negated NLI](#negated-nli)
    - [Commonsense Reasoning](#commonsense-reasoning)
    - [AI Generated](#ai-generated)
    - [Synthetic](#synthetic)
    - [Other task format that can be transformed to NLI format](#other-task-format-that-can-be-transformed-to-nli-format)
  - [Approaches and Applications](#approaches-and-applications)
    - [Symbolic](#symbolic)
    - [Data Extension](#data-extension)
  - [Datasets](#datasets)
    - [Question Answering](#question-answering)
    - [Natural Language Inference](#natural-language-inference)
    - [Test Suites](#test-suites)
  - [Models](#models)
  - [Benchmarking](#benchmarking)
  - [Resources](#resources)
    - [Repos](#repos)
    - [Workshops](#workshops)
- [NON-VERBAL REASONING](#non-verbal-reasoning)
  - [Video](#video)
  - [Image](#image)


## ✨VERBAL REASONING
### ✨Surveys
1. **Towards LOGIGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2310.00836) 2024. Mar. 

2. **Towards Reasoning in Large Language Models: A Survey**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.67) 2023. July. 

3. **A Survey of Reasoning with Foundation Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2312.11562) 2023. Dec. 

4. **Logical Reasoning over Natural Language as Knowledge Representation: A Survey**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2303.12023) 2023. Mar. 

5. **Natural Language Reasoning, A Survey**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2303.14725v2) 2023. May. 

6. **SymbolicAnd Neural Approaches to Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://scholarworks.iu.edu/iuswrrest/api/core/bitstreams/cec63ddb-9930-4fbe-b1f6-5e5a60503d5d/content) 2021. Jun. 

7. **A Survey on Recognizing Textual Entailment as an NLP Evaluation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.eval4nlp-1.10) 2020. Nov. 

8. **Recent Advanced in Natural Language Inferenece: A Survey of Benchmarks, Resources, and Approahces**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1904.01172) 2019. Apr. 

9. **An Overview of Natural Language Inference Data Collection: The Way Forward?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W17-7203) 2017. Oct. 

10. **Logical Formalization of Commonsense Reasoning: A Survey**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://www.jair.org/index.php/jair/article/view/11076/26258) 2017. Aug. 

11. **A Survey of Paraphrasing and Textual Entailment Method**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://www.jair.org/index.php/jair/article/view/10651/25463) 2010. May. 

12. **Natural Language Inference A Dissertation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://www-nlp.stanford.edu/~wcmac/papers/nli-diss.pdf) 2009. Jun. 



---

### ✨Formal Logical Reasoning (Deductive Reasoning)
1. **Aligning with Logic: Measuring, Evaluating and Improving Logical Consistency in Large Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2410.02205) 2024. Oct. 

2. **Deductive Additivity for Planning of Natural Language Proofs**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.nlrse-1.11/) 2023. June. 

3. **Can Pretrained Language Models (Yet) Reason Deductively?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.eacl-main.106/) 2023. May. 

4. **A Generation-based Deductive Method for Math Word Problems**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.108/) 2023. Dec. 

5. **GeoDRL: A Self-Learning Framework for Geometry Problem Solving using Reinforcement Learning in Deductive Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.850/) 2023. July. 

6. **Hence, Socrates is mortal: A Benchmark for Natural Language Syllogistic Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.148/) 2023. July. 

7. **Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://proceedings.neurips.cc/paper_files/paper/2023/file/09425891e393e64b0535194a81ba15b7-Paper-Conference.pdf) 2023. 

8. **Deductive Verification of Chain-of-Thought Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://proceedings.neurips.cc/paper_files/paper/2023/file/72393bd47a35f5b3bee4c609e7bba733-Paper-Conference.pdf) 2023. 

9. **FaiRR: Faithful and Robust Deductive Reasoning over Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.acl-long.77/) 2022. May. 

10. **AnaLog: Testing Analytical and Deductive Logic Learnability in Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.starsem-1.5/) 2022. July. 

11. **RobustLR: A Diagnostic Benchmark for Evaluating Logical Robustness of Deductive Reasoners**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.653/) 2022. Dec. 

12. **Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.acl-long.410/) 2022. May. 

13. **Natural Language Deduction with Incomplete Information**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.564/) 2022. Dec. 

14. **Natural Language Deduction through Search over Statement Compositions**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-emnlp.358/) 2022. Dec. 

15. **Natural Language Deduction with Incomplete Information**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.564/) 2022. Dec. 

16. **Teaching Machine Comprehension with Compositional Explanations**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.findings-emnlp.145/) 2020. Nov. 

17. **Inductive and deductive inferences in a Crowdsourced Lexical-Semantic Network**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/R13-1096/) 2013. Sep. 

18. **Questions require an answer: A deductive perspective on questions and answers**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/U06-1016/) 2006. Nov. 



---

### ✨Informal Logical Reasoning
#### ✨Inductive Reasoning
1. **Language Models as Inductive Reasoners**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2024.eacl-long.13/) 2024. Mar. 

2. **A Comprehensive Evaluation of Inductive Reasoning Capabilities and Problem Solving in Large Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2024.findings-eacl.22/) 2024. March. 

3. **It is not True that Transformers are Inductive Learners: Probing NLI Models with External Negation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2024.eacl-long.116/) 2024. Mar. 

4. **Contrastive Learning for Inference in Dialogue**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.631/) 2023. Dec. 

5. **I2D2: Inductive Knowledge Distillation with NeuroLogic and Self-Imitation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.535/) 2023. July. 

6. **Contrastive Learning with Generated Representations for Inductive Knowledge Graph Embedding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.900/) 2023. July. 

7. **Query Structure Modeling for Inductive Logical Reasoning Over Knowledge Graphs**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.259/) 2023. July. 

8. **Inductive Relation Prediction with Logical Reasoning Using Contrastive Representations**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.286/) 2022. Dec. 

9. **Deep Inductive Logic Reasoning for Multi-Hop Reading Comprehension**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.acl-long.343/) 2022. May. 

10. **Flexible Generation of Natural Language Deductions**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.506/) 2021. Nov. 

11. **Learning Explainable Linguistic Expressions with Neural Inductive Logic Programming for Sentence Classification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.emnlp-main.345/) 2020. Nov. 

12. **Thinking Like a Skeptic: Defeasible Inference in Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.findings-emnlp.418) 2020. Nov. 

13. **CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D19-1458/) 2019. Nov. 

14. **Inductive and deductive inferences in a Crowdsourced Lexical-Semantic Network**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/R13-1096/) 2013. Sep. 

15. **Incorporating Linguistics Constraints into Inductive Logic Programming**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W00-0740/) 2000. 



---

#### ✨Abductive Reasoning
1. **Self-Consistent Narrative Prompts on Abductive Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.ijcnlp-main.67/) 2023. Nov. 

2. **Abductive Commonsense Reasoning Exploiting Mutually Exclusive Explanations**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.831/) 2023. July. 

3. **Multi-modal Action Chain Abductive Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.254/) 2023. July. 

4. **True Detective: A Deep Abductive Reasoning Benchmark Undoable for GPT-3 and Challenging for GPT-4**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.starsem-1.28/) 2023. July. 

5. **How well do SOTA legal reasoning models support abductive reasoning?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2304.06912) 2023. Apr. 

6. **Simple Augmentations of Logical Rules for Neuro-SymbolicKnowledge Graph Completion**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-short.23/) 2023. July. 

7. **Forward-Backward Reasoning in Large Language Models for Mathematical Verification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://openreview.net/forum?id=GhYXocT75t) 2023. Sep. 

8. **Are Large Language Models Really Good Logical Reasoners? A Comprehensive Evaluation and Beyond**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2306.09841) 2023. Jun. 

9. **Case-Based Abductive Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.coling-1.134/) 2022. Oct. 

10. **AbductionRules: Training Transformers to Explain Unexpected Inputs**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-acl.19/) 2022. May. 

11. **LAMBADA: Backward Chaining for Automated Reasoning in Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2212.13894) 2022. Dec. 

12. **ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.findings-acl.317) 2021. Aug. 

13. **Learning as Abduction: Trainable Natural Logic Theorem Prover for Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.starsem-1.3) 2020. Dec. 

14. **Abductive Commonsense Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://openreview.net/forum?id=Byg1v1HKDB) 2019. Dec. 

15. **Abductive Explanation-based Learning Improves Parsing Accuracy and Efficiency**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W03-1715) 



---

#### ✨Analogical Reasoning
1. **Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://luarxiv.org/abs/2404.12728) 2024. Apr. 

2. **ANALOGICAL - A Novel Benchmark for Long Text Analogy Evaluation in Large Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.218/) 2023. July. 

3. **Can language models learn analogical reasoning? Investigating training objectives and comparisons to human performance**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.1022/) 2023. Dec. 

4. **StoryAnalogy: Deriving Story-level Analogies from Large Language Models to Unlock Analogical Understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.706/) 2023. Dec. 

5. **In-Context Analogical Reasoning with Pre-Trained Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.109) 2023. July. 

6. **ThinkSum: Probabilistic reasoning over sets using large language models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.68/) 2023. July. 

7. **E-KAR: A Benchmark for Rationalizing Natural Language Analogical Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-acl.311/) 2022. May. 

8. **Analogical Math Word Problems Solving with Enhanced Problem-Solution Association**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.643/) 2022. Dec. 

9. **A Neural-SymbolicApproach to Natural Language Understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-emnlp.158/) 2022. Dec. 



---



---

### ✨Logical Fallacy
1. **Detecting Argumentative Fallacies in the Wild: Problems and Limitations of Large Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.argmining-1.1) 2023. Dec. 

2. **Argument-based Detection and Classification of Fallacies in Political Debates**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.684) 2023. Dec. 

3. **Multitask Instruction-based Prompting for Fallacy Recognition**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.560/) 2022. Dec. 

4. **The Search for Agreement on Logical Fallacy Annotation of an Infodemic**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.lrec-1.471/) 2022. Jun. 

5. **Logical Fallacy Detection**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-emnlp.532/) 2022. Dec. 

6. **Breaking Down the Invisible Wall of Informal Fallacies in Online Discussions**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.acl-long.53/) 2021. Aug. 



---

### ✨Argument
1. **Uncovering Implicit Inferences for Improved Relational Argument Mining**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.eacl-main.182/) 2023. May. 

2. **Implicit Premise Generation with Discourse-aware Commonsense Knowledge Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.504) 2021. Nov. 

3. **TakeLab at SemEval-2018 Task12: Argument Reasoning Comprehension with Skip-Thought Vectors**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/S18-1192) 2018. Jun. 

4. **The Argument Reasoning Comprehension Task: Identification and Reconstruction of Implicit Warrants**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/N18-1175/) 2018. Jun. 

5. **Automatically Identifying Implicit Arguments to Improve Argument Linking and Coherence Modeling**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/S13-1043) 2013. Jun. 



---

### ✨Inference
1. **NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.acl-long.554/) 2023. July. 

2. **LogicAttack: Adversarial Attacks for Evaluating Logical Consistency of Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-emnlp.889) 2023. Dec. 

3. **QA-NatVer: Question Answering for Natural Logic-based Fact Verification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.521) 2023. Dec. 

4. **Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.nlrse-1.12/) 2023. Jun. 

5. **Conditional Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-emnlp.456) 2023. Dec. 

6. **Neuro-SymbolicNatural Logic with Introspective Revision for Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.tacl-1.14/) 2022. 

7. **Decomposing Natural Logic Inferences for Neural NLI**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.blackboxnlp-1.33/) 2022. Dec. 

8. **PLOG: Table-to-Logic Pretraining for Logical Table-to-Text Generation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.373/) 2022. Dec. 

9. **Pragmatic and Logical Inferences in NLI Systems: The Case of Conjunction Buttressing**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.unimplicit-1.2/) 2022. July. 

10. **Logical Reasoning with Span-Level Predictions for Interpretable and Robust NLI Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.251/) 2022. Dec. 

11. **ProoFVer: Natural Logic Theorem Proving for Fact Verification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.tacl-1.59) 2022. 

12. **Neural Natural Logic Inference for Interpretable Question Answering**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.298/) 2021. Nov. 

13. **NeuralLog: Natural Language Inference with Joint Neural and Logical Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.starsem-1.7/) 2021. Aug. 

14. **Monotonic Inference for Underspecified Episodic Logic**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.naloma-1.5/) 2021. Jun. 

15. **A (Mostly) SymbolicSystem for Monotonic Inference with Unscoped Episodic Logical Forms**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.naloma-1.9/) 2021. Jun. 

16. **Neural Unification for Logic Reasoning over Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.findings-emnlp.331/) 2021. Nov. 

17. **Logical Inferences with Comparatives and Generalized Quantifiers**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.acl-srw.35/) 2020. July. 

18. **MonaLog: a Lightweight System for Natural Language Inference Based on Monotonicity**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.scil-1.40) 2020. 

19. **Natural Language Inference with Monotonicity**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W19-0502/) 2019. May. 

20. **Combining Natural Logic and Shallow Reasoning for Question Answering**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/P16-1042) 2016. Aug. 

21. **Knowledge-Guided Linguistic Rewrites for Inference Rule Verification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/N16-1011) 2016. Jun. 

22. **Higher-order logical inference with compositional semantics**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D15-1244/) 2015. Sep. 

23. **A Tableau Prover for Natural Logic and Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D15-1296) 2015. Sep. 

24. **NaturalLI: Natural Logic Inference for Common Sense Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D14-1059/) 2014. Oct. 

25. **NLog-like Inference and Commonsense Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2014.lilt-9.9/) 2014. 

26. **Natural Logic for Textual Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W07-1431/) 2007. Jun. 

27. **Semantic and Logical Inference Model for Textual Entailment**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W07-1418/) 2007. Jun. 

28. **Recognising Textual Entailment with Logical Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/H05-1079) 2005. Oct. 

29. **Augmenting Neural Networks with First-order Logic**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/P19-1028) 



---

### ✨Legal
1. **Syllogistic Reasoning for Legal Judgment Analysis**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.864) 2023. Dec. 

2. **How well do SOTA legal reasoning models support abductive reasoning?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2304.06912) 2023. Apr. 

3. **Can AMR Assist Legal and Logical Reasoning?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-emnlp.112) 2022. Dec. 

4. **From legal to technical concept: Towards an automated classification of German political Twitter postings as criminal offenses**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/N19-1135/) 2019. June. 

5. **Learning Logical Structures of Paragraphs in Legal Articles**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/I11-1003) 2011. Nov. 



---

### ✨Critical Thinking
1. **Critical Thinking for Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.iwcs-1.7) 2021. June. 



---

### ✨Theorem Proving
1. **LangPro: Natural Language Theorem Prover**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D17-2020/) 2017. Sep. 

2. **A Tableau Prover for Natural Logic and Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D15-1296) 2015. Sep. 



---

### ✨Natural Language Inference
1. **Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.nlrse-1.12/) 2023. Jun. 

2. **CURRICULUM: A Broad-Coverage Benchmark for Linguistic Phenomena in Natural Language Understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.naacl-main.234) 2022. July. 

3. **Entailment as Few-Shot Learners**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2104.14690) 2021. Apr. 

4. **GLUE: A Multi-task Benchmark and Analysis Platform for Natural Language Understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1804.07461v3) 2019. Feb. 

5. **Inherent Disagreement in Human Textual Inferences**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/Q19-1043) 2019. May. 

6. **A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MNLI)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1704.05426v4.pdf) 2018. Jun. 

7. **Lessons from Natural Language Inference in the Clinical Domain**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1808.06752) 2018. Aug. 

8. **A Large Annotated Corpus for Learning Language Inference (SNLI)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://nlp.stanford.edu/pubs/snli_paper.pdf) 2015. Sept. 

9. **SemEval-2014 Task 1: Evaluzation of Compositional Distributional Semantic Models on Full Sentence through Semantic Relatedness and Textual Entailment (SICK)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf) 2014. Aug. 

10. **The Winograd Schema Challenge**

	[![](https://img.shields.io/badge/📄-Paper-orange)](http://commonsensereasoning.org/2011/papers/Levesque.pdf) 2011. 

11. **The Fourth PASCAL Recognizing Textual Entailment Challenge**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://tac.nist.gov/publications/2008/additional.papers/RTE-4_overview.proceedings.pdf) 2008. 

12. **The Third PASCAL Recognizing Textual Entailment Challenge**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W07-1401) 2007. 

13. **Natural Logic for Textual Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://nlp.stanford.edu/pubs/natlog-wtep07.pdf) 2007. Jun. 

14. **The Second PASCAL Recognizing Textual Entailment Challenge**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://www.semanticscholar.org/paper/The-Second-PASCAL-Recognising-Textual-Entailment-Bar-Haim-Dagan/136326377c122560768db674e35f5bcd6de3bc40) 2006. 

15. **The PASCAL Recognizing Textual Entailment Challenge**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://kdd.cs.ksu.edu/Courses/Fall-2008/CIS798/Handouts/06-dagan05pascal.pdf) 2005. 

16. **Using the Framework (FraCaS)**

	1996. 

#### ✨Multi-Choice Question
1. **BiQuAD: Towards QA based on deeper text understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.starsem-1.10/) 2021. Aug. 

2. **StrategyQA**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering) 



---

#### ✨Converted from QA
1. **Transforming Question Answering Datasets Into Natural Language Inference Datasets**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://www.semanticscholar.org/reader/8f1c9b656157b1d851563fb42129245701d83175) 2018. Sep. 

2. **Reading Comprehension as Natural Language Inference: A Semantic Analysis**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.starsem-1.2) 2012. Dec. 



---

#### ✨Converted from Summarization
1. **Falsesum: Generating Document-level NLI Examples for Recognizing Factual Inconsistency in Summerization**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.naacl-main.199) 2022. Jul. 

2. **SUMMAC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.tacl-1.10) 2021. Aug. 



---

#### ✨Tabular Premises (Tabular Reasoning)
1. **SemEval-2021 Task 9: Fact Verification and Evidence Finding for Tabular Data in Scientific Documents (SEM-TAB-FACTS)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.semeval-1.39) 2021. Aug. 

2. **Is My Model Using the Right Evidence? Systematic Probes for Examining Evidence-Based Tabular Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.tacl-1.38) 2021. Sep. 

3. **The Fact Extraction and VERification Over Unstructured and Structured information (FEVEROUS) Shared Task**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.fever-1.1) 2021. Nov. 

4. **INFOTABS: Inference on Tables as Semi-structured Data**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.acl-main.210) 2020. Jul. 

5. **TABFACT: A Large-scale Dataset for Table-base Fact Verification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1909.02164v5.pdf) 2019. Sep. 



---

#### ✨Fact-checking
1. **FEVER: A Large-scale Dataset for Fact Extraction and VERification**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/N18-1074) 2018. Jul. 

2. **Fake News Challenge**

	[![](https://img.shields.io/badge/🌐-Webpage-blue)](http://www.fakenewschallenge.org/) 2016. Dec. 



---

#### ✨Probabilistic NLI
1. **Uncertain Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1909.03042) 2019. Sep. 

2. **A Probabilistic Classification Approach for Lexical Textual Entailment**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://cdn.aaai.org/AAAI/2005/AAAI05-166.pdf) 2005. 



---

#### ✨Document-level NLI
1. **Falsesum: Generating Document-level NLI Examples for Recognizing Factual Inconsistency in Summarization**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.naacl-main.199) 2022. Jul. 

2. **BioNLI: Generating a Biomedical NLI Dataset Using Lexico-semantic Constraints for Adversarial Examples**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2210.14814) 2022. Oct. 

3. **Validity Assessment of Legal Will Statement as Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2210.16989v1.pdf) 2022. Oct. 

4. **LawngNLI: A Multigranular, Long-premise NLI Benchmark for Evaluating Models’ In-domain Generalization from Short to Long Contexts**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-emnlp.369) 2022. Dec. 

5. **DOCNLI: A Large-scale Dataset for Document-level Natural Language Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.findings-acl.435) 2021. Aug. 

6. **ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.findings-emnlp.164) 2021. Nov. 

7. **Evaluating the Factual Consistency of Abstractive Text Summarization (FactCC)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.emnlp-main.750) 2020. Nov. 

8. **Natural Language Inference in Context — Investigating Contextual Reasoning over Long Texts**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2011.04864) 2011. Nov. 



---

#### ✨Complex Reasoning NLI
1. **Explaining Answers with Entailment Trees (ENTAILMENTBANK)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://www.semanticscholar.org/reader/4a56f72b9c529810ba4ecfe9eac522d87f6db81d) 2022. May. 

2. **FOLIO: Natural Language Reasoning with First-Order-Logic**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2209.00840) 2022. Sep. 

3. **Diagnosing the First-Order Logical Reasoning Ability Through LogicNLI**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.303) 2021. Nov. 

4. **Transformers as Soft Reasoners over Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2002.05867) 2020. Feb. 

5. **Adversarial NLI: A New Benchmark for Natural Language Understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.acl-main.441) 2020. Jul. 

6. **Are Natural Language Inference Models Impressive? Learning Implicature and Presupposition**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.acl-main.768) 2020. Jul. 

7. **Natural Language Inference in Context — Investigating Contextual Reasoning over Long Texts**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2011.04864) 2020. Nov. 

8. **ConjNLI: Natural Language Inference Over Conjunctive Sentences**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.emnlp-main.661) 2020. Nov. 

9. **TaxiNLI: Taking a ride up the NLU hill**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.conll-1.4) 2020. Nov. 

10. **Can neural networks understand monotonicity reasoning? (MED)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/W19-4804v2) 2019. 

11. **HELP: A Dataset for Identifying Shortcomings of Neural Models in Monotonicity Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/S19-1027) 2019. Jun. 

12. **Collecting Diverse Natural Language Inference Problems for Sentence Representation Evaluation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D18-1007) 2018. Oct. 



---

#### ✨Negated NLI
1. **Not another Negation Benchmark: The NaN-NLI Test Suite for Sub-clausal Negation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.aacl-main.65) 2022. Nov. 

2. **Neural Natural Language Inference Models Partially Embed Theories of Lexical Entailment and Negation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2004.14623v4.pdf) 2020. Apr. 

3. **An Analysis of Natural Language Inference Benchmarks through the Lens of Negation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.emnlp-main.732) 2020. Nov. 



---

#### ✨Commonsense Reasoning
1. **The Winograd Schema Challenge**

	[![](https://img.shields.io/badge/📄-Paper-orange)](http://commonsensereasoning.org/2011/papers/Levesque.pdf) 2011. 



---

#### ✨AI Generated
1. **WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2201.05955) 2022. Jan. 



---

#### ✨Synthetic
1. **Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (HANS)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1902.01007) 2019. Feb. 



---

#### ✨Other task format that can be transformed to NLI format
1. **Explaining Answers with Entailment Trees (ENTAILMENT BANK)**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.585/) 2021. Nov. 

2. **FraCaS: Temporal Analysis**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2012.10668) 2020. Dec. 



---



---

### ✨Approaches and Applications
#### ✨Symbolic
1. **Enhancing Ethical Explanations of Large Language Models through Iterative SymbolicRefinement**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2024.eacl-long.1/) 2024. March. 

2. **Improved Logical Reasoning of Language Models via Differentiable SymbolicProgramming**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.191) 2023. July. 

3. **Logic-LM: Empowering Large Language Models with SymbolicSolvers for Faithful Logical Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-emnlp.248/) 2023. Dec. 

4. **Improved Logical Reasoning of Language Models via Differentiable SymbolicProgramming**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.191/) 2023. July. 

5. **LINC: A NeuroSymbolicApproach for Logical Reasoning by Combining Language Models with First-Order Logic Provers**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.313/) 2023. Dec. 

6. **Investigating Transformer guided Chaining for Interpretable Natural Logic Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.588) 2023. July. 

7. **Analytical, Symbolicand First-Order Reasoning within Neural Architectures**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.cstfrs-1.7) 2021. July. 

8. **Conversational Multi-Hop Reasoning with Neural Commonsense Knowledge and SymbolicLogic Rules**

	[![](https://img.shields.io/badge/📄-Paper-orange)](http://arxiv.org/abs/2109.08544) 2021. Sept. 

9. **Learning SymbolicRules for Reasoning in Quasi-Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](http://arxiv.org/abs/2111.12038) 2021. Nov. 

10. **Are Pretrained Language Models SymbolicReasoners over Knowledge?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://doi.org/10.18653/v1/2020.conll-1.45) 2020. 

11. **Differentiable Reasoning on Large Knowledge Bases and Natural Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://doi.org/10.1609/aaai.v34i04.5962) 2020. Apr. 



---

#### ✨Data Extension
1. **MERIt: Meta-Path Guided Contrastive Learning for Logical Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2203.00357) 2022. Mar. 

2. **Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2105.03659) 2021. May. 



---



---

### ✨Datasets
#### ✨Question Answering
1. **LogiQA2.0 - An Improved Dataset for Logic Reasoning in Question Answering and Textual Inference**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://ieeexplore.ieee.org/abstract/document/10174688) 2023. May. 

2. **ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://openreview.net/pdf?id=HJgJtT4tvB) [![](https://img.shields.io/badge/🌐-Webpage-blue)](https://whyu.me/reclor/) 2023. 

3. **MTR: A Dataset Fusing Inductive, Deductive, and Defeasible Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.640/) 2023. July. 

4. **AR-LSAT: Investigating Analytical Reasoning of Text**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2104.06598) [![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/zhongwanjun/AR-LSAT) 2021. Apr. 

5. **Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering) [![](https://img.shields.io/badge/🌐-Webpage-blue)](https://allenai.org/data/strategyqa) 2021. 

6. **CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/1908.06177) 2019. Sep. 



---

#### ✨Natural Language Inference
1. **FOLIO: Natural Language Reasoning with First-Order Logic**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2209.00840) [![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/Yale-LILY/FOLIO) 2022. 

2. **Diagnosing the First-Order Logical Reasoning Ability Through LogicNLI**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.303/) [![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/omnilabNLP/LogicNLI) 2021. 

3. **Transformers as Soft Reasoners over Language**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2002.05867) 2020. Feb. 



---

#### ✨Test Suites
1. **GLoRE: Evaluating Logical Reasoning of Large Language Models.**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2310.09107) [![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/csitfun/glore) 2023. Oct. 

2. **LogicBench: A Benchmark for Evaluation of Logical Reasoning.**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://openreview.net/forum?id=7NR2ZVzZxx) 2023. Dec. 

3. **Hence, Socrates is mortal: A Benchmark for Natural Language Syllogistic Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.findings-acl.148/) 2023. July. 

4. **True Detective: A Deep Abductive Reasoning Benchmark Undoable for GPT-3 and Challenging for GPT-4**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.starsem-1.28/) 2023. July. 

5. **StoryAnalogy: Deriving Story-level Analogies from Large Language Models to Unlock Analogical Understanding**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.706/) 2023. Dec. 

6. **How susceptible are LLMs to Logical Fallacies?**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2308.09853) 2023. Aug. 

7. **RobustLR: A Diagnostic Benchmark for Evaluating Logical Robustness of Deductive Reasoners**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.emnlp-main.653/) 2022. Dec. 

8. **AbductionRules: Training Transformers to Explain Unexpected Inputs**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-acl.19/) 2022. May. 

9. **E-KAR: A Benchmark for Rationalizing Natural Language Analogical Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2022.findings-acl.311/) 2022. May. 

10. **RuleBERT: Teaching Soft Rules to Pre-Trained Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2021.emnlp-main.110/) 2021. Nov. 

11. **PuzzLing Machines: A Challenge on Learning From Small Data**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2020.acl-main.115/) 2020. July. 

12. **CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/D19-1458/) 2019. Nov. 

13. **Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2310.00836) [![](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/datasets/logicreasoning/logi_glue) 



---



---

### ✨Models
1. **LLaMA-7B-LogiCoT**

	[![](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/csitfun/llama-7b-logicot) 

2. **LOGIPT**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2311.06158) 

3. **Symbol-LLM**

	[![](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/Symbol-LLM) 



---

### ✨Benchmarking
1. **Evaluating the Logical Reasoning Ability of ChatGPT and GPT-4**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2304.03439) 2023. Apr. 

2. **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2302.04023) 2023. Nov. 

3. **Are Large Language Models Really Good Logical Reasoners? A Comprehensive Evaluation and Beyond**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://arxiv.org/abs/2306.09841) 2021. March. 



---

### ✨Resources
#### ✨Repos
1. **LogiTorch**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/LogiTorch/logitorch) a PyTorch-based library for logical reasoning on natural language

2. **GLoRE**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/csitfun/glore) a benchmark for evaluating the logical reasoning of LLMs

3. **Logiformer**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/xufangzhi/logiformer) This is a model to tackle the logical reasoning task in the field of multiple-choice machine reading comprehension. The code of the decoder part is not the final version, but it is one of the alternatives. You can also implement it based on own design, which may further improve the experimental results.

4. **Awesome Natural Language Reasoning Papers**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/mengzaiqiao/awesome-natural-language-reasoning) A collection of research papers related to Knowledge Reasoning with Natural Language Models.

5. **Awesome LLM Reasoning**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/atfortes/Awesome-LLM-Reasoning) Curated collection of papers and resources on how to unlock the reasoning ability of LLMs and MLLMs.

6. **Deep-Reasoning-Papers**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/floodsung/Deep-Reasoning-Papers) Recent Papers including Neural SymbolicReasoning, Logical Reasoning, Visual Reasoning, natural language reasoning and any other topics connecting deep learning and reasoning.

7. **Awesome deep logic**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/ccclyu/awesome-deeplogic) Must-Read Papers or Resources on how to integrate Symboliclogic into deep neural nets.

8. **Awesome-Reasoning-Foundation-Models**

	[![](https://img.shields.io/badge/📦-Github-purple)](https://github.com/reasoning-survey/Awesome-Reasoning-Foundation-Models) Papers on reasoning fundation models.



---

#### ✨Workshops
1. **Workshop on Natural Language Reasoning and Structured Explanations (2023)**

	



---



---



---

## ✨NON-VERBAL REASONING
### ✨Video
1. **Multimodal Fallacy Classification in Political Debates**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2024.eacl-short.16/) 2024. March. 

2. **ART: rule bAsed futuRe-inference deducTion**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/2023.emnlp-main.592/) 2023. Dec. 



---

### ✨Image
1. **Emergent Communication for Rules Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8ace30c68b085556ccce04ed4ae4ebb-Paper-Conference.pdf) 2023. 

2. **A Benchmark for Compositional Visual Reasoning**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://proceedings.neurips.cc/paper_files/paper/2022/file/c08ee8fe3d19521f3bfa4102898329fd-Paper-Datasets_and_Benchmarks.pdf) 2022. 

3. **Multimodal Logical Inference System for Visual-Textual Entailment**

	[![](https://img.shields.io/badge/📄-Paper-orange)](https://aclanthology.org/P19-2054/) 2019. July. 

4. **The Abstraction and Reasoning Corpus (ARC)**

	[![](https://img.shields.io/badge/🌐-Webpage-blue)](https://github.com/fchollet/ARC) 



---



---

