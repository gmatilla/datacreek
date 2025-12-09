from enum import Enum


class TaskStatus(str, Enum):
    """Enumerate statuses recorded during long-running tasks."""

    INGESTING = "ingesting"
    GENERATING = "generating"
    CLEANUP = "cleanup"
    EXPORTING = "exporting"
    SAVING_NEO4J = "saving_neo4j"
    LOADING_NEO4J = "loading_neo4j"
    SAVING_REDIS_GRAPH = "saving_redis_graph"
    LOADING_REDIS_GRAPH = "loading_redis_graph"
    DELETING = "deleting"
    EXTRACTING_FACTS = "extracting_facts"
    EXTRACTING_ENTITIES = "extracting_entities"
    OPERATION = "operation"
    COMPLETED = "completed"
    FAILED = "failed"
