# Benchmark performances

hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1

|                 Tasks                 |Version|Filter|n-shot|Metric|Value |   |Stderr|
|---------------------------------------|-------|------|-----:|------|-----:|---|-----:|
|mmlu                                   |N/A    |none  |     0|acc   |0.4259|±  |0.1068|
| - humanities                          |N/A    |none  |     0|acc   |0.4060|±  |0.1082|
|  - formal_logic                       |Yaml   |none  |     0|acc   |0.2302|±  |0.0376|
|  - high_school_european_history       |Yaml   |none  |     0|acc   |0.5576|±  |0.0388|
|  - high_school_us_history             |Yaml   |none  |     0|acc   |0.6078|±  |0.0343|
|  - high_school_world_history          |Yaml   |none  |     0|acc   |0.5907|±  |0.0320|
|  - international_law                  |Yaml   |none  |     0|acc   |0.5785|±  |0.0451|
|  - jurisprudence                      |Yaml   |none  |     0|acc   |0.5278|±  |0.0483|
|  - logical_fallacies                  |Yaml   |none  |     0|acc   |0.5337|±  |0.0392|
|  - moral_disputes                     |Yaml   |none  |     0|acc   |0.4740|±  |0.0269|
|  - moral_scenarios                    |Yaml   |none  |     0|acc   |0.2425|±  |0.0143|
|  - philosophy                         |Yaml   |none  |     0|acc   |0.4887|±  |0.0284|
|  - prehistory                         |Yaml   |none  |     0|acc   |0.5278|±  |0.0278|
|  - professional_law                   |Yaml   |none  |     0|acc   |0.3246|±  |0.0120|
|  - world_religions                    |Yaml   |none  |     0|acc   |0.6374|±  |0.0369|
| - other                               |N/A    |none  |     0|acc   |0.5043|±  |0.0909|
|  - business_ethics                    |Yaml   |none  |     0|acc   |0.4300|±  |0.0498|
|  - clinical_knowledge                 |Yaml   |none  |     0|acc   |0.4604|±  |0.0307|
|  - college_medicine                   |Yaml   |none  |     0|acc   |0.3468|±  |0.0363|
|  - global_facts                       |Yaml   |none  |     0|acc   |0.4100|±  |0.0494|
|  - human_aging                        |Yaml   |none  |     0|acc   |0.5381|±  |0.0335|
|  - management                         |Yaml   |none  |     0|acc   |0.5631|±  |0.0491|
|  - marketing                          |Yaml   |none  |     0|acc   |0.7009|±  |0.0300|
|  - medical_genetics                   |Yaml   |none  |     0|acc   |0.4200|±  |0.0496|
|  - miscellaneous                      |Yaml   |none  |     0|acc   |0.6488|±  |0.0171|
|  - nutrition                          |Yaml   |none  |     0|acc   |0.4477|±  |0.0285|
|  - professional_accounting            |Yaml   |none  |     0|acc   |0.3262|±  |0.0280|
|  - professional_medicine              |Yaml   |none  |     0|acc   |0.4007|±  |0.0298|
|  - virology                           |Yaml   |none  |     0|acc   |0.4277|±  |0.0385|
| - social_sciences                     |N/A    |none  |     0|acc   |0.4748|±  |0.0928|
|  - econometrics                       |Yaml   |none  |     0|acc   |0.2544|±  |0.0410|
|  - high_school_geography              |Yaml   |none  |     0|acc   |0.5101|±  |0.0356|
|  - high_school_government_and_politics|Yaml   |none  |     0|acc   |0.5907|±  |0.0355|
|  - high_school_macroeconomics         |Yaml   |none  |     0|acc   |0.3692|±  |0.0245|
|  - high_school_microeconomics         |Yaml   |none  |     0|acc   |0.3193|±  |0.0303|
|  - high_school_psychology             |Yaml   |none  |     0|acc   |0.5615|±  |0.0213|
|  - human_sexuality                    |Yaml   |none  |     0|acc   |0.5420|±  |0.0437|
|  - professional_psychology            |Yaml   |none  |     0|acc   |0.4101|±  |0.0199|
|  - public_relations                   |Yaml   |none  |     0|acc   |0.5182|±  |0.0479|
|  - security_studies                   |Yaml   |none  |     0|acc   |0.4367|±  |0.0318|
|  - sociology                          |Yaml   |none  |     0|acc   |0.7015|±  |0.0324|
|  - us_foreign_policy                  |Yaml   |none  |     0|acc   |0.6400|±  |0.0482|
| - stem                                |N/A    |none  |     0|acc   |0.3308|±  |0.0859|
|  - abstract_algebra                   |Yaml   |none  |     0|acc   |0.2800|±  |0.0451|
|  - anatomy                            |Yaml   |none  |     0|acc   |0.4296|±  |0.0428|
|  - astronomy                          |Yaml   |none  |     0|acc   |0.4408|±  |0.0404|
|  - college_biology                    |Yaml   |none  |     0|acc   |0.4306|±  |0.0414|
|  - college_chemistry                  |Yaml   |none  |     0|acc   |0.2100|±  |0.0409|
|  - college_computer_science           |Yaml   |none  |     0|acc   |0.2900|±  |0.0456|
|  - college_mathematics                |Yaml   |none  |     0|acc   |0.2600|±  |0.0441|
|  - college_physics                    |Yaml   |none  |     0|acc   |0.2451|±  |0.0428|
|  - computer_security                  |Yaml   |none  |     0|acc   |0.5700|±  |0.0498|
|  - conceptual_physics                 |Yaml   |none  |     0|acc   |0.3574|±  |0.0313|
|  - electrical_engineering             |Yaml   |none  |     0|acc   |0.4207|±  |0.0411|
|  - elementary_mathematics             |Yaml   |none  |     0|acc   |0.2566|±  |0.0225|
|  - high_school_biology                |Yaml   |none  |     0|acc   |0.4548|±  |0.0283|
|  - high_school_chemistry              |Yaml   |none  |     0|acc   |0.2759|±  |0.0314|
|  - high_school_computer_science       |Yaml   |none  |     0|acc   |0.3900|±  |0.0490|
|  - high_school_mathematics            |Yaml   |none  |     0|acc   |0.2556|±  |0.0266|
|  - high_school_physics                |Yaml   |none  |     0|acc   |0.2450|±  |0.0351|
|  - high_school_statistics             |Yaml   |none  |     0|acc   |0.2315|±  |0.0288|
|  - machine_learning                   |Yaml   |none  |     0|acc   |0.3214|±  |0.0443|

|      Groups      |Version|Filter|n-shot|Metric|Value |   |Stderr|
|------------------|-------|------|-----:|------|-----:|---|-----:|
|mmlu              |N/A    |none  |     0|acc   |0.4259|±  |0.1068|
| - humanities     |N/A    |none  |     0|acc   |0.4060|±  |0.1082|
| - other          |N/A    |none  |     0|acc   |0.5043|±  |0.0909|
| - social_sciences|N/A    |none  |     0|acc   |0.4748|±  |0.0928|
| - stem           |N/A    |none  |     0|acc   |0.3308|±  |0.0859|
