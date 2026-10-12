# Dicionário e Classificação dos Dados

## Classificação

| Coluna | Descrição | Tipo | Uso | Classificação de privacidade | Controle recomendado |
|---|---|---|---|---|---|
| `person_age` | idade da pessoa | numérico | feature | dado pessoal/quase-identificador | faixa válida, minimização e análise por grupos |
| `person_income` | renda anual | numérico | feature | dado pessoal financeiro | acesso restrito, criptografia e limites válidos |
| `person_home_ownership` | situação da moradia | categórico | feature | dado pessoal socioeconômico | categorias controladas e avaliação de proxy |
| `person_emp_length` | tempo de emprego | numérico | feature | dado pessoal profissional | imputação documentada e faixa válida |
| `loan_intent` | finalidade do empréstimo | categórico | feature | dado pessoal financeiro | categorias controladas e finalidade limitada |
| `loan_grade` | classe de risco do empréstimo | categórico | feature | perfil de crédito | rastreabilidade e explicação do critério |
| `loan_amnt` | valor solicitado | numérico | feature | dado pessoal financeiro | valor positivo e acesso restrito |
| `loan_int_rate` | taxa de juros | numérico | feature | condição contratual/financeira | faixa válida e monitoramento de drift |
| `loan_status` | inadimplência observada | binário | alvo | perfil de crédito | domínio 0/1, acesso restrito e retenção limitada |
| `loan_percent_income` | parcela da renda comprometida | numérico | feature | dado financeiro derivado | faixa válida e avaliação de proporcionalidade |
| `cb_person_default_on_file` | inadimplência anterior registrada | categórico | feature | histórico de crédito | necessidade, exatidão, explicação e acesso restrito |
| `cb_person_cred_hist_length` | duração do histórico de crédito | numérico | feature | histórico de crédito | faixa válida e retenção controlada |
| `prediction` | classe prevista | binário | saída | decisão/perfil derivado | explicabilidade, revisão e auditoria |
| `prediction_probability` | probabilidade estimada | numérico | saída | score derivado | proteção contra exposição e interpretação responsável |

## Dados não coletados pelo projeto

O dataset acadêmico não contém nome, CPF, e-mail, telefone, endereço completo,
biometria ou geolocalização precisa. Em produção, esses dados devem permanecer
fora do conjunto de features, salvo necessidade demonstrada e aprovação formal.

## Qualidade e proveniência

- fonte: OpenML, dataset 43454;
- finalidade: pesquisa acadêmica de classificação binária financeira;
- referência: registros limpos e validados antes do treino;
- produção: amostra simulada com drift controlado;
- contrato: schema Pandera com domínio, faixas, nulidade e duplicidade;
- rastreabilidade: versão do código, semente e artefatos registrados por execução.

## Observação sobre dados sensíveis

Os atributos financeiros merecem proteção elevada, mas não devem ser chamados
automaticamente de dados pessoais sensíveis na definição estrita da LGPD. Uma
implantação real deve verificar se outras fontes introduzem origem racial ou
étnica, religião, opinião política, filiação sindical, saúde, vida sexual, genética
ou biometria.
