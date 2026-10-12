# Plano de Privacidade e Proteção de Dados

## 1. Escopo

Este documento descreve os controles de privacidade aplicáveis ao projeto acadêmico
`home-credit-observability`. Em uma implantação real, o plano deve ser revisado
pelo encarregado de dados, pelas áreas jurídica, de segurança e de risco da
instituição financeira.

O dataset OpenML 43454 não contém nome, CPF, endereço, telefone ou e-mail. Mesmo
assim, atributos como idade, renda, histórico de crédito e condições do empréstimo
podem ser dados pessoais ou quase-identificadores quando associados a uma pessoa
em um ambiente real.

## 2. Finalidade e necessidade

As finalidades são:

- avaliar risco de inadimplência em solicitações de crédito;
- monitorar qualidade, drift e desempenho do modelo;
- investigar falhas, alertas e decisões contestadas;
- manter evidências de auditoria e governança do modelo.

Somente atributos necessários para essas finalidades devem ser coletados. Nome,
CPF, endereço completo e dados de contato não devem integrar as features do
modelo. Identificadores operacionais devem ser substituídos por chaves aleatórias
e mantidos em ambiente separado.

## 3. Base legal

A base legal principal proposta para o tratamento necessário à análise de risco é
a proteção do crédito, prevista no art. 7º, X, da LGPD. Procedimentos preliminares
relacionados a contrato podem também se aplicar, conforme o art. 7º, V.

Consentimento não é adotado como base legal genérica para a decisão de crédito.
Tratamentos secundários, como marketing ou enriquecimento não necessário à
operação, exigem avaliação própria de finalidade, necessidade e base legal.

## 4. Agentes e responsabilidades

| Papel | Responsabilidade |
|---|---|
| Controlador | definir finalidades, base legal, retenção e critérios da decisão |
| Operador | tratar dados somente conforme instruções documentadas |
| Encarregado | receber solicitações dos titulares e orientar a governança |
| Responsável pelo modelo | validar desempenho, drift, vieses e versões |
| Segurança | controlar acessos, criptografia, incidentes e evidências |
| Risco/Compliance | aprovar limites, exceções, revisão humana e auditorias |

## 5. Privacy by Design e by Default

- prevenção de riscos antes da implantação;
- minimização de dados e features;
- privacidade como configuração padrão;
- separação entre identificadores e atributos analíticos;
- criptografia em trânsito e em repouso;
- acesso pelo menor privilégio;
- logs sem PII, tokens ou segredos;
- rastreabilidade do dataset, modelo, configuração e decisão;
- transparência sobre critérios e uso dos dados;
- revisão periódica de necessidade, retenção, drift e vieses.

## 6. Ciclo de vida e retenção

Os prazos abaixo são uma política proposta para o projeto, não prazos legais
universais. Em produção, devem ser conciliados com obrigações regulatórias e com a
política corporativa.

| Categoria | Prazo proposto | Destinação ao final |
|---|---:|---|
| cache do dataset bruto acadêmico | 90 dias | exclusão; pode ser baixado novamente |
| dataset de referência pseudonimizado | duração da versão ativa + 180 dias | exclusão ou anonimização |
| registros individuais de inferência | 180 dias | agregação e exclusão dos registros individuais |
| logs técnicos sem PII | 12 meses | exclusão segura |
| relatórios agregados de qualidade e drift | 24 meses | arquivamento ou exclusão |
| modelo, métricas, configurações e model card | 5 anos | revisão e descarte controlado |
| dados sob investigação ou obrigação legal | até o encerramento aplicável | descarte autorizado por Jurídico/Compliance |

Uma rotina trimestral deve localizar dados vencidos, registrar a eliminação e
preservar apenas o necessário para obrigação legal, defesa de direitos ou
auditoria devidamente justificada.

## 7. Segurança

- TLS para transferências e criptografia no armazenamento;
- RBAC e autenticação multifator nos ambientes de produção;
- separação de desenvolvimento, homologação e produção;
- segredos em cofre, nunca no Git ou nos logs;
- backups criptografados e testes periódicos de restauração;
- inventário de acessos e revisão trimestral de permissões;
- validação Pandera antes da inferência;
- registro de versão do modelo, dataset e configuração;
- plano de resposta a incidente com contenção, investigação, comunicação e lições aprendidas.

## 8. Direitos dos titulares e decisões automatizadas

O canal do encarregado deve permitir confirmação de tratamento, acesso, correção,
informação sobre compartilhamento, oposição quando aplicável e demais direitos da
LGPD. Para decisões exclusivamente automatizadas, o titular pode solicitar revisão
e informações claras sobre critérios e procedimentos.

Uma contestação de crédito deve gerar protocolo, preservar as evidências da
decisão, encaminhar o caso a pessoa qualificada e comunicar o resultado em
linguagem compreensível. Segredo comercial não deve ser usado para eliminar a
transparência exigida pela LGPD.

## 9. Vieses e não discriminação

- atributos sensíveis não devem ser usados sem necessidade e fundamento válidos;
- métricas devem ser avaliadas por grupos quando houver base legal e controles;
- proxies potenciais de características protegidas devem ser investigados;
- quedas de recall e diferenças entre grupos devem gerar revisão humana;
- retreinamento depende de aprovação, documentação e comparação com o modelo atual;
- decisões adversas devem ser explicáveis e contestáveis.

## 10. Compartilhamento e terceiros

Compartilhamentos devem ter finalidade documentada, minimização, contrato com
obrigações de segurança, registro do operador e mecanismo de exclusão ao término.
Dados reais não devem ser enviados para notebooks públicos, serviços pessoais ou
ferramentas não aprovadas.

## 11. Incidentes

O fluxo mínimo é: detectar, conter, preservar evidências, avaliar impacto,
notificar os responsáveis internos, corrigir a causa e documentar a decisão sobre
comunicação à ANPD e aos titulares. Cada incidente deve produzir ações preventivas
e atualização dos controles.

## 12. Evidências de conformidade

- contrato de dados e relatórios de falha;
- hashes ou versões dos datasets;
- métricas e alertas de monitoramento;
- inventário de acessos;
- registro de retenção e descarte;
- model card e análise causal;
- histórico de versões e aprovações;
- registro de solicitações e revisões humanas.

## 13. Limitações

Este projeto usa dados públicos para fins educacionais e não concede crédito real.
O plano demonstra controles esperados, mas não substitui avaliação jurídica,
RIPD, testes de segurança ou validação regulatória de uma implantação financeira.

## 14. Referências

- BRASIL. Lei nº 13.709/2018 - Lei Geral de Proteção de Dados Pessoais.
- ANPD. Direitos dos titulares de dados pessoais.
- CAVOUKIAN, Ann. Privacy by Design: The 7 Foundational Principles.
