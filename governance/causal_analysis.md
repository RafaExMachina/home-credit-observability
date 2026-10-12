# Análise Causal da Degradação

## Pergunta

As mudanças observadas em renda, taxa de juros e comprometimento de renda causaram
a redução do recall do modelo?

## Evidências observadas

- PSI e Kolmogorov-Smirnov detectaram drift em `person_income`, `loan_int_rate` e
  `loan_percent_income`;
- seis dos quatorze testes estatísticos sinalizaram mudança;
- o recall caiu de 0,7771 para 0,6823;
- o ROC AUC caiu de 0,8712 para 0,8662;
- o cenário de produção foi criado por transformação controlada dos dados.

## Hipótese causal

Uma mudança econômica pode alterar condições de crédito e perfil dos solicitantes.
Isso modifica a distribuição das features e pode quebrar relações aprendidas pelo
modelo, aumentando falsos negativos e reduzindo o recall.

```text
Condições econômicas
      |------------> taxa de juros --------|
      |------------> renda ----------------|--> perfil de risco --> inadimplência
      |------------> demanda/seleção ------|
                                      |
Política de concessão ----------------|

Distribuição de treino --> parâmetros do modelo --> predição
Mudança de população ---------------------------> erro/degradação
```

## Confundidores e vieses de seleção

- desemprego, inflação e região;
- mudança na política de aprovação;
- campanhas e canais de aquisição;
- mix de produtos e valores solicitados;
- exclusão de clientes antes de entrarem no dataset;
- atraso ou erro na observação de inadimplência.

## Limites da conclusão

PSI e KS demonstram associação temporal e mudança de distribuição, não causalidade.
Como o drift foi artificialmente introduzido, o experimento demonstra que a
transformação escolhida acompanha degradação, mas não prova que inflação, juros ou
renda seriam a causa em produção real.

## Investigação recomendada

- acompanhar coortes antes e depois da mudança;
- registrar política de crédito e origem do cliente;
- analisar falsos negativos por segmento;
- usar janelas temporais e testes de robustez;
- estimar efeitos somente após explicitar o grafo causal e pressupostos;
- realizar teste prospectivo controlado quando ética e operacionalmente possível;
- submeter conclusões e intervenções à revisão de Risco e Compliance.

## Prescrição

Manter revisão humana para casos adversos, elevar monitoramento dos três atributos,
investigar segmentos afetados e comparar recalibração com retreinamento. Nenhuma
ação deve ser tomada apenas porque uma feature apresentou drift.
