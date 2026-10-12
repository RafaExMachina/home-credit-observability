# Model Card - Credit Risk Baseline

## Visão geral

| Campo | Valor |
|---|---|
| Modelo | Regressão Logística |
| Versão documentada | 0.3.0 |
| Problema | classificação binária de risco de inadimplência |
| Dataset | OpenML 43454 |
| Alvo | `loan_status` |
| Saídas | classe e probabilidade de inadimplência |
| Uso | demonstração acadêmica de qualidade, drift e observabilidade |

## Finalidade pretendida

Demonstrar o ciclo de treinamento, validação, monitoramento e governança de um
modelo de risco. O artefato não está aprovado para concessão real de crédito.

## Uso proibido

- decisão real sem validação jurídica, de risco e de segurança;
- decisão adversa sem possibilidade de revisão;
- uso de dados para finalidade incompatível;
- inferência de características sensíveis;
- exposição pública de scores individuais;
- substituição automática de análise humana em casos contestados.

## Desempenho

| Métrica | Baseline de teste | Referência pontuada | Produção simulada |
|---|---:|---:|---:|
| Accuracy | 0,8059 | 0,8114 | 0,8498 |
| Precision | 0,5389 | 0,5487 | 0,6443 |
| Recall | 0,7821 | 0,7771 | 0,6823 |
| F1-score | 0,6381 | 0,6432 | 0,6627 |
| ROC AUC | 0,8716 | 0,8712 | 0,8662 |

A produção simulada apresentou queda de aproximadamente 9,48 pontos percentuais
no recall. A acurácia maior não compensa a redução na detecção de inadimplentes.

## Riscos e limitações

- dados públicos podem não representar a população-alvo da instituição;
- o cenário de produção é simulado;
- correlação e drift não demonstram causalidade;
- regressão logística pode não capturar relações não lineares;
- features socioeconômicas podem atuar como proxies de grupos protegidos;
- métricas globais podem ocultar disparidades entre subgrupos;
- probabilidade estimada não deve ser interpretada isoladamente como verdade.

## Controles

- contrato Pandera antes da inferência;
- separação entre referência e produção;
- PSI, KS e relatório Evidently;
- alertas para drift, recall, ROC AUC, nulos e latência;
- logs estruturados sem identificadores diretos;
- revisão humana para decisões contestadas;
- versionamento do modelo e dos artefatos;
- aprovação formal antes de promover uma nova versão.

## Critérios de revisão

O modelo deve ser investigado quando ocorrer drift, quebra do contrato, queda de
recall superior a 0,05, ROC AUC inferior a 0,80, disparidade relevante entre
grupos ou mudança de política de crédito. Retreinamento não é automático: exige
análise de causa, validação comparativa e aprovação.
