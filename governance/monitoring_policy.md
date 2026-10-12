# Política de Monitoramento e Resposta

## Objetivo

Definir métricas, limites, responsabilidades e ações para detectar degradação sem
permitir que um alerta isolado promova automaticamente um novo modelo.

## Indicadores e limites

| Indicador | Limite | Severidade | Ação inicial |
|---|---:|---|---|
| predições nulas | maior que 0% | critical | bloquear lote e investigar schema/pipeline |
| latência p95 | maior que 200 ms | warning | verificar infraestrutura e tamanho do lote |
| features com drift | maior que 0 | warning | identificar variáveis, segmentos e origem |
| delta de recall | menor que -0,05 | critical | revisão do modelo e da política de decisão |
| ROC AUC | menor que 0,80 | critical | suspender promoção e investigar degradação |
| quebra do contrato | qualquer ocorrência | critical | bloquear ingestão |

## Frequência

- por lote: qualidade, nulidade, volume e schema;
- por execução: latência, drift, métricas e alertas;
- semanal: tendências e reincidência de alertas;
- mensal: desempenho por segmentos e avaliação de vieses;
- trimestral: acesso, retenção, model card e necessidade das features.

## Responsabilidades

| Evento | Responsável | Prazo inicial proposto |
|---|---|---:|
| lote bloqueado | Engenharia de Dados | 4 horas |
| falha crítica de desempenho | Responsável pelo Modelo e Risco | 1 dia útil |
| suspeita de discriminação | Risco, Compliance e Encarregado | imediato |
| incidente de privacidade | Segurança e Encarregado | imediato |
| drift sem queda de desempenho | Ciência de Dados | 3 dias úteis |

## Fluxo de resposta

1. registrar o alerta e preservar evidências;
2. confirmar se é falha de dados, infraestrutura, política ou modelo;
3. segmentar métricas e comparar com a referência;
4. avaliar impacto sobre titulares e grupos;
5. aplicar mitigação temporária, inclusive revisão humana;
6. decidir entre correção de dados, recalibração, retreinamento ou rollback;
7. validar a solução com testes e aprovação independente;
8. documentar causa, decisão e ação preventiva.

## Promoção e rollback

Uma nova versão só pode substituir a atual quando contratos, testes, métricas,
fairness, segurança e documentação forem aprovados. O artefato anterior deve ser
mantido durante a janela de rollback. Alertas nunca devem disparar retreinamento e
implantação sem validação humana.

## Escopo das medições corrigidas

Latência é de chamadas individuais ao serviço já carregado, após warm-up, incluindo pré-processamento, sem HTTP. O Evidently usa as mesmas sete features numéricas, a mesma política PSI >= 0,25 OU KS p < 0,05 e limiar global de pelo menos uma feature. A comparação de recall usa a mesma coorte independente antes/depois. O experimento sintético de Concept Drift monitora desempenho com rótulos disponíveis, sem medir latência.
