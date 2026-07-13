# Alfredo: SDK open source y benchmark de memoria para agentes

**Fecha:** 2026-07-13  
**Estado:** Diseño aprobado  
**Repositorio:** `E:\CODE\Alfredo`

## Objetivo

Evolucionar Alfredo desde una submission funcional de hackathon hacia un proyecto open source reutilizable y ampliamente distribuible. La prioridad es un SDK de memoria para desarrolladores, acompañado por un benchmark público que demuestre memoria longitudinal, seguridad y eficiencia de contexto. La plataforma gestionada queda fuera de la primera etapa y se habilitará únicamente cuando existan señales de adopción.

La meta de 100k estrellas se trata como objetivo de distribución, no como garantía técnica. Las decisiones de ingeniería deben optimizar utilidad, instalación, claridad, extensibilidad y confianza.

## Decisión de producto

Se adopta la combinación:

1. **SDK open source primero:** API Python estable, SQLite local por defecto, MCP, CLI, conectores LLM y providers intercambiables.
2. **Benchmark de referencia:** evaluación reproducible contra historial bruto y RAG sin lifecycle.
3. **Plataforma gestionada después:** HTTP multiusuario, dashboard, autenticación, observabilidad y backends gestionados solo tras validar adopción.

La experiencia objetivo es:

- demostrar valor en menos de cinco minutos;
- integrarse en menos de quince minutos;
- permitir migrar de local a producción sin romper la API pública.

## Alcance de la primera etapa

### Incluido

- estabilización de la fachada `MemoryAgent`;
- contratos pequeños para store, embeddings, retrieval y políticas de confianza;
- aislamiento explícito por usuario o namespace;
- ciclo completo de extracción, validación, almacenamiento, retrieval, trust filtering, context packing, refuerzo, supersession, decay y archivado;
- memorias explicables con score, señales, confianza y razón de aceptación o rechazo;
- SQLite como backend predeterminado;
- providers de embeddings reemplazables;
- adapters CLI, MCP, HTTP y LLM sin duplicación del dominio;
- benchmark con baselines y métricas de calidad, seguridad, coste de contexto y latencia;
- instalación y quickstart reproducibles en Windows, macOS y Linux;
- documentación de privacidad, seguridad, límites y extensión;
- tests de contratos, aislamiento, lifecycle, integraciones y regresiones.

### Fuera de alcance inicial

- servicio SaaS multi-tenant;
- dashboard web completo;
- autenticación y facturación;
- migración obligatoria a PostgreSQL o vector database;
- SDKs oficiales en varios lenguajes;
- promesas de rendimiento a escala no medida;
- mecanismos diseñados para inflar artificialmente estrellas o métricas.

## Arquitectura

### Fachada pública

`MemoryAgent` será la entrada estable para los consumidores. Debe permitir inyección de dependencias sin obligar al usuario a conocer SQLite o `sentence-transformers`.

```python
agent = MemoryAgent(
    store=SQLiteStore(".alfredo/memory.db"),
    embedder=SentenceTransformerEmbedder(),
)

result = agent.run(
    user_id="demo-user",
    session_id="session-1",
    message="Prefiero respuestas breves y en español.",
)
```

### Componentes

- **Memory model:** hechos, preferencias, episodios, procedimientos y estados.
- **Extraction:** candidatos con confianza, origen y sensibilidad.
- **Storage:** SQLite inicial y protocolo para futuros backends.
- **Embedding:** backend local configurable y tolerante a fallos.
- **Retrieval:** señales semánticas, léxicas, temporales, importancia, confianza, strength y diversidad.
- **Trust policy:** expiración, supersession, forget explícito, sensibilidad y prompt injection.
- **Lifecycle:** reinforce, consolidate, decay, archive y restore.
- **Context packer:** límite explícito de tokens o bytes y trazabilidad de selección.
- **Adapters:** CLI, MCP, HTTP y conectores LLM.

Los componentes dependen de interfaces pequeñas y no de implementaciones concretas. El orquestador no debe conocer detalles de SQLite ni del proveedor de embeddings.

## Modelo de memoria

Cada memoria activa debe poder representar:

- identificador estable;
- `user_id` o namespace;
- tipo;
- contenido;
- confianza;
- importancia;
- sensibilidad;
- origen;
- timestamps de creación, actualización, acceso y expiración;
- estado: activa, supersedida, archivada u olvidada;
- relación de supersession;
- número de recuperaciones;
- razón de la última recuperación.

Las memorias archivadas no se recuperan por defecto. Las memorias sensibles requieren una política explícita antes de entrar en el contexto del modelo.

## Flujo de datos

```text
Observe
  -> Extract
  -> Validate
  -> Store
  -> Retrieve
  -> Trust-filter
  -> Pack bounded context
  -> Generate response
  -> Reinforce useful memories
  -> Supersede contradictions
  -> Decay stale memories
  -> Explain decision
```

El flujo debe conservar trazabilidad suficiente para que una respuesta pueda asociarse con las memorias consultadas, las señales de ranking y las decisiones de confianza.

Ejemplo de resultado explicable:

```json
{
  "memory_id": "mem_123",
  "score": 0.87,
  "matched_by": ["semantic", "recent", "important"],
  "trust": "accepted",
  "reason": "current preference supersedes older preference"
}
```

## Diferenciación

Alfredo no se posicionará como un wrapper de historial ni como un buscador vectorial. La propuesta es una memoria con lifecycle:

- almacena lo útil;
- recupera lo relevante;
- filtra lo inseguro o desactualizado;
- refuerza lo que sigue siendo útil;
- supersede contradicciones;
- olvida o archiva lo obsoleto;
- entrega un paquete de contexto acotado y explicable.

## Validación

### Experiencia de instalación

- instalación limpia en Windows, macOS y Linux;
- primera memoria almacenada en menos de cinco minutos;
- demo local sin API key;
- SQLite y modo determinista o fallback ligero para la primera ejecución;
- comandos copiables sin configuración oculta.

### Contratos de ingeniería

Se cubrirán:

- persistencia y migraciones;
- aislamiento entre usuarios;
- actualización y supersession;
- forget explícito;
- expiración;
- sensibilidad;
- resistencia a prompt injection;
- límites de contexto;
- providers intercambiables;
- compatibilidad MCP;
- errores de embeddings sin pérdida de la memoria textual.

### Benchmark

Alfredo se comparará contra:

1. historial bruto completo;
2. RAG semántico sin lifecycle;
3. Alfredo con trust policy y forgetting.

Métricas mínimas:

- recall temporal;
- precisión de la preferencia actual;
- contradicción resuelta;
- abstención de baja confianza;
- filtrado de memorias olvidadas;
- filtrado de datos sensibles;
- resistencia a instrucciones maliciosas;
- tokens enviados al modelo;
- latencia p50 y p95;
- uso de memoria;
- tamaño del vault.

### Adopción

Se observarán tiempo hasta primera ejecución, demos completadas, errores de instalación, integraciones, contribuciones externas, issues repetidas y releases reproducibles. Las estrellas son una señal de distribución, no el criterio único de calidad.

## Riesgos y mitigaciones

- **Parecer otro proyecto RAG:** diferenciar con lifecycle, trust policy, supersession, forgetting y benchmark comparativo.
- **Acoplamiento a SQLite o embeddings locales:** usar protocolos y dependency injection desde la API pública.
- **Memorias inseguras en prompts:** aplicar trust filtering antes de context packing y probar ataques sintéticos.
- **Complejidad excesiva para nuevos usuarios:** conservar defaults locales y un quickstart mínimo.
- **Resultados de benchmark no creíbles:** publicar baselines, dataset, comandos, versión y métricas reproducibles.
- **Alcance de plataforma demasiado temprano:** no implementar SaaS hasta contar con señales de adopción.

## Criterios de aceptación del diseño

El diseño se considera implementable cuando:

- un desarrollador puede usar `MemoryAgent` sin conocer la implementación del store;
- los adapters comparten el mismo dominio y no duplican lifecycle;
- un usuario puede inspeccionar por qué una memoria fue aceptada, ignorada o archivada;
- el benchmark compara alternativas y reporta calidad, seguridad, coste y latencia;
- una futura migración a backend gestionado no requiere romper contratos públicos;
- la documentación permite instalar, ejecutar y validar el proyecto sin credenciales externas.

## Próxima fase

Crear un plan de implementación por etapas, empezando por contratos y experiencia de instalación, continuando con lifecycle explicable y benchmark comparativo, y terminando con integraciones, documentación y verificación.
