Revise e consolide o MVP de benchmarking de `expf` em GPUs NVIDIA já existente neste fork privado do LLVM, em `/home/leandro/projects/github/leandrolcampos/llvm-project`. Leia as instruções locais e preserve alterações preexistentes. Não recrie a infraestrutura.

Os itens 1–8 definem os requisitos. A avaliação do código existente e as correções prioritárias estão no item 9. Preserve a definição da medida, os algoritmos e a seleção padrão da libc ao corrigir os achados.

1. **Escopo**

   Use a RTX 5070 como plataforma inicial e permita selecionar a arquitetura explicitamente quando necessário: por exemplo, para H100 e L4. Meça a vazão sustentada de avaliações escalares de `expf` em um kernel computacional controlado, com entradas em registradores e GPU ocupada. A definição operacional e os critérios para declarar saturação estão no item 7.

   Candidatos iniciais:

   - LLVM-libc `float_eval::expf`;
   - LLVM-libc `double_eval::expf`, com accurate path;
   - LLVM-libc `double_eval::expf`, com `LIBC_MATH_SKIP_ACCURATE_PASS`;
   - LLVM-libc `float_eval::cr_expf_rn`;
   - libdevice `__nv_expf`;
   - libdevice `__nv_fast_expf`, correspondente a `__expf` neste Clang.

   Uma implementação com configurações diferentes constitui candidatos diferentes. Futuras variantes `cr_expf_tbN_poly` seguem o mesmo modelo de candidatos LLVM-libc: `N` é o número de entradas da tabela e `poly` é o método de avaliação, como `horner` ou `estrin`.

   Preserve os algoritmos e a seleção padrão da libc. Novas implementações matemáticas, latência serial, domínios com resultados não finitos e certificação de arredondamento ficam fora do MVP.

2. **Build**

   Crie `gpu-libm-bench/` e integre-o ao build iniciado por `cmake -S llvm -B build` usando:

   ```text
   LLVM_EXTERNAL_PROJECTS=gpu-libm-bench
   LLVM_EXTERNAL_GPU_LIBM_BENCH_SOURCE_DIR=<caminho>
   ```

   Mantenha compatibilidade com `LLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld`, `LLVM_ENABLE_RUNTIMES=openmp;offload;libc` e `LLVM_RUNTIME_TARGETS=default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda`. Encaminhe os launchers de compilação, como ccache, aos subbuilds aplicáveis.

   Use um subbuild nativo para o runner e NVBench. Compile os kernels medidos com o Clang/LLVM do build selecionado; a toolchain do runner deve ser compatível com o CUDA Toolkit. Estabeleça dependências respeitando a criação dos targets de runtimes.

   Separe compilação de execução. Configuração e compilação devem funcionar sem GPU. Desabilitar o projeto deve remover seus requisitos de CUDA/NVBench.

3. **NVBench**

   Incorpore os fontes de uma revisão fixada de NVBench em `third-party/nvbench`, preservando as licenças. Não use submódulo nem download durante configuração ou compilação.

   Crie `third-party/update_nvbench.sh` seguindo `third-party/update_boost_math.sh`: versão/revisão explícita, download e extração, remoção dos metadados de VCS e limpeza do arquivo temporário. Preserve os arquivos necessários ao build.

   Prepare também as dependências transitivas necessárias em versões fixadas e com fontes locais. O script de atualização pode obtê-las; o CMake deve consumi-las sem acesso à rede.

   Integre os targets de NVBench ao subbuild nativo. Compile-o automaticamente como dependência do benchmark, sem instalação manual ou no sistema. Desabilite exemplos, testes de dispositivo e profiling não utilizados.

   NVBench é o executor obrigatório das medições: registro dos benchmarks, eixos de parâmetros, seleção do dispositivo, aquecimento, repetição dos lançamentos, eventos CUDA e exportação. O runner carrega os cubins compilados por Clang e fornece seus lançamentos a NVBench; não implemente um loop de timing paralelo ao framework. Use a API da revisão vendorizada, conforme `third-party/nvbench/docs/benchmarks.md`.

4. **Registro de candidatos**

   Use JSON com apenas os campos necessários:

   ```json
   [
     {
       "id": "llvm-double-skip",
       "provider": "llvm-libc",
       "cmake": {
         "LIBC_CONF_MATH_OPTIMIZATIONS": "LIBC_MATH_ASSUME_ROUND_NEAREST_ONLY | LIBC_MATH_NO_ERRNO | LIBC_MATH_NO_EXCEPT | LIBC_MATH_SKIP_ACCURATE_PASS"
       }
     },
     {
       "id": "nvidia-expf",
       "provider": "libdevice",
       "symbol": "__nv_expf"
     }
   ]
   ```

   A interface é `float(float)`. Candidatos LLVM-libc exportam `expf`; derive seus targets e caminhos do build. Configure o caminho da libdevice uma vez para o projeto.

   Cada mapa `cmake` deve conter a configuração matemática completa e eventuais opções de compilação específicas do candidato, incluindo contração quando aplicável. Acrescentar um candidato já selecionável por CMake deve exigir apenas seu registro. Runner e kernels não devem enumerar implementações.

5. **Bitcode e seleção**

   Gere o bitcode de cada candidato LLVM-libc em um subbuild isolado, reutilizando `libmbitcode`. Passe o mapa `cmake` como opções `-D`, que devem prevalecer sobre `libc/config/gpu/config.json`. Não modifique esse JSON.

   Confira o seletor em `libc/src/__support/math/expf.h`. Para `LIBC_CONF_MATH_OPTIMIZATIONS`, utilize como base:

   ```text
   LIBC_MATH_ASSUME_ROUND_NEAREST_ONLY | LIBC_MATH_NO_ERRNO | LIBC_MATH_NO_EXCEPT
   ```

   | Implementação | Opções adicionais à base |
   |---|---|
   | double com accurate path | Nenhuma |
   | double sem accurate path | `LIBC_MATH_SKIP_ACCURATE_PASS` |
   | float | `LIBC_MATH_INTERMEDIATE_COMP_IN_FLOAT \| LIBC_MATH_SKIP_ACCURATE_PASS` |
   | cr | `LIBC_MATH_INTERMEDIATE_COMP_IN_FLOAT` |

   Armazene as expressões completas no registro. Nos dois candidatos double, mantenha desabilitados `LIBC_MATH_INTERMEDIATE_COMP_IN_FLOAT` e `LIBC_MATH_SMALL_TABLES`. Verifique os valores efetivos e a implementação selecionada.

   Se necessário, adicione o menor seletor CMake específico de `expf`, mantendo a seleção automática como padrão. Não crie seletores para algoritmos inexistentes.

   Preserve os produtos de cada candidato na árvore de build, sem substituir bibliotecas instaladas nem recompilar LLVM/Clang por candidato. Confirme os símbolos NVIDIA e o mapeamento de `__expf` nos headers.

   Use a mesma cadeia final por candidato e arquitetura: kernel + bitcode → ligação e preparação do IR → otimização → PTX → ptxas → cubin. Preserve `NVVMReflect`, permita inlining e selecione explicitamente a função NVIDIA. O runner pode carregar os cubins pela CUDA Driver API; não use chamadas indiretas no dispositivo.

6. **Inputs e kernel**

   Gere uma vez no host um corpus em `[-10,10]`, usando semente configurável de 64 bits, padrão 1. Reutilize o algoritmo de `RandomGenerator` em `libc/benchmarks/gpu/Random.h`: SplitMix64 para inicialização e xorshift64*, preservando constantes e operações inteiras.

   Para cada índice crescente `j`, calcule:

   ```text
   k = rng.next32() >> 8
   u = double(k) * 0x1p-24
   x[j] = float(-10.0 + 20.0 * u)
   ```

   Use RN ties-to-even. Essa transformação define uma grade uniforme de 24 bits na escala real, seguida de arredondamento para binary32. Os bits de entrada devem independer do candidato, da ordem de execução e da amostragem adaptativa.

   Defina o kernel por `B` blocos, `T` threads/bloco, `K` fluxos/thread e `R` avaliações/fluxo. Todas as `B*T` threads executam as `K*R` avaliações; não inclua threads inativas na contagem. Cada thread carrega `K` valores-base do corpus antes do loop, mantém seus estados em registradores e grava os resultados ou um checksum ao terminar. Não faça cargas dos inputs globais nem PRNG a cada avaliação. Cargas de tabelas e demais acessos próprios da implementação de `expf` permanecem no trecho medido.

   Antes da coleta comparativa, fixe no código e no README o mapeamento `(thread, fluxo, iteração) → argumento`, eventual reutilização do corpus, operações auxiliares, dependências e saídas observáveis. O mecanismo antiotimização deve preservar exatamente `R` avaliações por fluxo; repetir uma função pura com argumento invariável não basta. Se variar argumentos ou introduzir dependências, preserve a sequência numérica entre candidatos e descreva essas operações. Use o mesmo workload para todos, mudando apenas a implementação selecionada.

   Verifique no código otimizado que as avaliações não foram eliminadas ou movidas para fora do loop e que os estados não foram materializados em arrays de memória local. Fluxos podem conter dependências internas; a independência requerida é entre os `K` fluxos e entre threads. Cada lançamento reinicializa o estado e reutiliza inputs imutáveis, sem depender dos resultados do lançamento anterior.

7. **Medição e resultados**

   A unidade de trabalho é uma avaliação escalar `float → float`, por thread. Para uma configuração `theta=(B,T,K,R)` e candidato `c`, defina:

   ```text
   C = B * T * K * R                         # avaliações por lançamento
   t = soma dos tempos GPU dos lotes / soma dos lançamentos medidos
   Q_kernel(c, theta) = C / (1e9 * t)         # Gchamadas/s; t em segundos
   ```

   Calcule `C` em 64 bits, sem overflow, e confirme sua correspondência com o código executado, mesmo após inlining. Uma instrução de warp não equivale a uma avaliação escalar.

   A estimativa é condicionada à GPU, toolchain, configuração, regime de cache e corpus utilizados. A incerteza temporal com uma semente fixa não mede a variabilidade entre corpora; generalizações sobre a distribuição exigem sementes independentes.

   Use como tempo principal o resultado **Batch GPU** de NVBench: tempo de eventos no stream cobrindo lançamentos consecutivos, dividido pelo número de lançamentos. Na API atual, corresponde a `nv/batch/time/gpu/mean`; confira esse identificador na revisão fixada. Os lotes medidos excluem o aquecimento. Medidas Cold, se produzidas pelo framework, são diagnósticas e não entram em `Q_kernel`.

   Para cada estado de NVBench, selecione o cubin e os parâmetros fora de `state.exec`, registre `C` com `state.add_element_count(C)` e use `state.exec(nvbench::exec_tag::gpu, launcher)`. O `launcher` deve executar somente um `cuLaunchKernel`, usando `launch.get_stream()` e o contexto CUDA do dispositivo selecionado. Não faça sincronização, cópia, reset, alocação ou carregamento de módulo nesse callback. Represente candidato e `B,T,K,R` por eixos de NVBench; derive os valores de candidatos do registro.

   Obtenha `Q_kernel` explicitamente do tempo Batch GPU. Não use uma coluna automática de elementos/s baseada em outro modo de timing. Registre os parâmetros de amostragem, número de lançamentos e eventual timeout; resultados cujo orçamento de coleta não foi atendido devem ser marcados como incompletos.

   O intervalo inclui todo o trabalho do kernel: avaliações, controle, mecanismo antiotimização, cargas iniciais e stores finais. O lote também pode incluir intervalos entre lançamentos no stream. Geração do corpus, transferências host/dispositivo e preparação ficam fora. A medida é vazão do workload definido, não tempo exclusivo, latência ou máximo intrínseco de `expf`. Não subtraia baseline nem converta `1/Q_kernel` em latência.

   O regime principal é aquecido: repita os mesmos módulos e inputs sem limpeza explícita de caches entre os lançamentos do lote. Isso não pressupõe que todas as tabelas caibam em cache. Minimize outros workloads GPU e registre política de clocks, frequências observadas e condições térmicas; não normalize resultados pela frequência nominal.

   Faça um piloto para fixar workload, grade de configurações, duração de coleta e tolerância de equivalência `epsilon` antes da comparação final. Use `epsilon=3%` como padrão configurável, declarado como critério operacional, não como constante do hardware. Para `T,K` fixos, exija que duas duplicações consecutivas de `R` mantenham `Q_kernel` equivalente, demonstrando amortização dos custos fixos; depois faça o mesmo com `B`, demonstrando um platô de concorrência. Considere equivalentes razões de vazão cujo IC de 95% esteja contido em `[1-epsilon,1+epsilon]`. Sem evidência suficiente, marque `saturação não demonstrada`.

   Varra também poucos valores de `T,K` e preserve todos os pontos medidos. Publique comparações na mesma configuração; se apresentar configurações ajustadas por candidato, identifique-as e selecione-as no piloto, confirmando-as com medições finais independentes. Um platô não prova que somente a computação de `expf` limita a vazão: verifique memória, spills e instruções auxiliares antes dessa atribuição. Profiling e inspeção de código são separados do timing.

   Colete pelo menos dez rodadas externas. Cada rodada percorre os candidatos e pontos finais em ordem permutada por uma semente registrada, produzindo uma estimativa Batch GPU por ponto. Mantenha o pareamento por rodada entre candidatos e configurações. Apresente mediana de `Q_kernel`, dispersão e IC de 95% da mediana por bootstrap percentil, reamostrando rodadas inteiras. Para razões, use a razão entre medianas e preserve esse pareamento. Fixe e registre a semente e o número de reamostragens. O IC é uma estimativa de incerteza, sem garantia de cobertura exata com dez rodadas. As amostras internas de NVBench e as lanes não são rodadas independentes; o critério de parada interno não substitui esse IC.

   Exporte uma tabela com ID, GPU/arquitetura, semente e tamanho do corpus, `B,T,K,R,C`, modo de timing, tempo Batch GPU, Gchamadas/s, número de lançamentos, rodada/ordem e status da medição. Identifique as versões das ferramentas e os parâmetros do protocolo no README ou na saída. Limite os artefatos aos produtos normais do build, à tabela e ao README; dumps e relatórios adicionais servem apenas à inspeção.

8. **Validação e entrega**

   Verifique:

   - os seis candidatos executam e resolvem para os símbolos e configurações esperados;
   - opções CMake prevalecem sobre o JSON GPU;
   - a mesma semente reproduz os bits de entrada, independentemente da ordem dos candidatos;
   - o workload está definido, fixado e preserva a sequência de inputs entre candidatos;
   - avaliações e dependências necessárias sobrevivem à otimização e correspondem a `C`;
   - o callback lança o kernel no stream de NVBench e a vazão usa exclusivamente o tempo Batch GPU;
   - um smoke test numérico, fora do timing, passa com referência independente e tolerâncias documentadas;
   - um candidato LLVM-libc já selecionável pode ser acrescentado apenas pelo registro;
   - amortização e platô atendem ao critério declarado, ou a saturação é marcada como não demonstrada;
   - resultados, incerteza e limitações correspondem à grandeza operacional definida, sem alegar custo exclusivo de `expf`.

   Documente os comandos de configuração, build, execução e atualização de NVBench, as versões das ferramentas e como registrar futuras variantes. Se alguma validação estiver impedida pelo ambiente, informe o impedimento e os comandos pendentes.

9. **Avaliação da implementação — terceira revisão de 2026-10-16**

   **Conclusão.** Os dois achados da segunda revisão foram corrigidos. A revisão do código, da integração com o coletor e das regressões não identificou novos defeitos substanciais no escopo examinado. Não há novas correções obrigatórias propostas; preserve os requisitos dos itens 1–8 e não amplie o MVP para encerrar esta revisão.

   **P1 resolvido — Falhas de smoke persistentes.** `gpu-libm-bench/runner/Validation.h` registra sucesso somente após a verificação terminar e memoriza a exceção original em caso de falha. Em `gpu-libm-bench/runner/main.cu`, os módulos só são publicados após o smoke numérico passar. A reprovação numérica bloqueia o candidato naquele dispositivo; a reprovação do workload bloqueia o candidato/K naquele dispositivo. Estados posteriores dependentes da verificação reprovada não entram no timing, preservam a causa, exportam `failed` e produzem código 1. O coletor mantém as linhas já persistidas; pontos sem todas as rodadas válidas permanecem incompletos na análise. Skips de política do NVBench continuam classificados separadamente como coleta incompleta.

   **P2 resolvido — Release instalada do driver.** O runner consulta `nvmlSystemGetDriverVersion` e exporta `nvidia_driver_version`, mantendo `cuda_driver_api_version` como identificação separada da versão de CUDA suportada pelo driver. A release participa do preflight, da conferência anterior ao timing, das linhas exportadas e da comparação piloto/final. O coletor exige uma string não vazia tanto no preflight quanto no protocolo do piloto; campos ausentes ou inválidos não são aceitos como identificação equivalente. As consultas permanecem restritas à execução, preservando configuração e compilação sem GPU.

   **Garantias anteriores preservadas.** Continuam presentes a publicação atômica e invalidação do manifesto em builds interrompidos, a verificação dos cubins sob lock durante a coleta, a persistência progressiva, a invalidação do cache pela configuração completa e identidade da toolchain, o encaminhamento de launchers e a conferência do UUID. O workload, o PRNG, o pipeline de bitcode, a leitura de Batch GPU e o bootstrap pareado mantêm a definição operacional. As verificações estruturais de IR e o smoke de checksum não são prova automática da contagem após inlining: permanece obrigatória a inspeção do cubin por candidato/arquitetura/K, vinculada ao seu hash. Locks e registros de inspeção têm os limites documentados: exclusão cooperativa e responsabilidade do revisor.

   **Validação disponível.** Na implementação dos ajustes, passaram seis testes host focados: rejeição persistente após falhas injetadas nos smokes, isolamento das chaves de validação, preservação dos resultados anteriores e propagação de falhas, confirmação com release igual, rejeição com release diferente apesar da mesma versão de API CUDA, rejeição de release ausente/inválida e manutenção dos caminhos existentes de coleta parcial e identidade da GPU. O runner foi recompilado e a consulta `--device-info=0` retornou `cuda_driver_api_version=13020` e `nvidia_driver_version=595.99.02` na RTX 5070. Esta terceira revisão examinou os fontes e essas evidências; não reexecutou testes, builds ou campanhas GPU. As regressões host exercitam o cache com falhas injetadas e o coletor com runner simulado; não representam injeção de falhas nos kernels reais.

   **Proveniência das campanhas anteriores.** `build/gpu-libm-bench-validation-20261016/pilot.csv` e `final.csv` preservam 2160 e 540 estados completos, respectivamente, em dez rodadas por campanha, para os seis candidatos. Esses dados antecedem os dois ajustes recentes e mantêm seus próprios hashes de runner/análise; o campo histórico `driver_version` contém a versão da API CUDA, sem release instalada coletada automaticamente. Não apresentá-los como dados do runner atualizado, não completar retroativamente metadados e não utilizá-los como `--pilot` de uma nova confirmação com esse runner. A tabela histórica de 60 estados também mantém sua identidade original.

   Na confirmação anterior, em B=384,T=256,K=4,R=1024, os quatro critérios de equivalência foram atendidos para `llvm-cr`, `llvm-double` e `llvm-double-skip`; para `llvm-float` e os dois candidatos libdevice, permanece `saturação não demonstrada`. Esses resultados descrevem o workload e o protocolo registrados, sem demonstrar um gargalo exclusivo de `expf`.

   **Limites e pendências experimentais.** Cada fluxo alterna apenas `-base,+base`; o tempo inclui checksum e controle. A correção do accurate path double para bits `0xc236bd8c` fica fora de `[-10,10]` e não é executada nesse domínio. Smoke não certifica arredondamento, uma semente não mede variabilidade entre corpora e platô não identifica sozinho o recurso limitante. H100/L4 ainda não foram executadas. Esses limites não exigem novas funcionalidades nem repetição das campanhas para validar as correções desta revisão.
