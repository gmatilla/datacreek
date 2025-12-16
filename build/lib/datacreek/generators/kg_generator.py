import logging
from typing import Any, Dict, List, Optional

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency
    KMeans = None

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.qa import QAPair
from datacreek.utils.config import get_prompt

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class KGGenerator(BaseGenerator):
    """Generate QA examples from a knowledge graph."""

    def _select_facts(self, kg: KnowledgeGraph, num: int) -> List[str]:
        """Return ``num`` fact IDs chosen to maximize coverage."""

        fact_nodes = [
            n for n, d in kg.graph.nodes(data=True) if d.get("type") == "fact"
        ]
        if len(fact_nodes) <= num:
            return fact_nodes

        texts = [
            f"{kg.graph.nodes[f]['subject']} {kg.graph.nodes[f]['predicate']} {kg.graph.nodes[f]['object']}"
            for f in fact_nodes
        ]
        embeddings = kg.index.transform(texts)
        if embeddings.size == 0:
            fact_nodes.sort(key=lambda n: kg.graph.degree(n), reverse=True)
            return fact_nodes[:num]

        if KMeans is None:
            raise ImportError("scikit-learn is required for KGGenerator")

        kmeans = KMeans(n_clusters=num, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(embeddings)
        selected: List[str] = []
        for cluster_id in range(num):
            members = [f for f, l in zip(fact_nodes, labels) if l == cluster_id]
            if not members:
                continue
            members.sort(key=lambda n: kg.graph.degree(n), reverse=True)
            selected.append(members[0])
        return selected

    def process_graph(
        self,
        kg: KnowledgeGraph,
        num_pairs: int = 25,
        *,
        verbose: bool = False,
        multi_answer: bool = False,
    ) -> Dict[str, Any]:
        """Generate QA pairs by sampling facts from ``kg``.

        Facts are selected using clustering to maximise coverage. For each
        chosen fact (``subject``-``predicate``-``object`` triple) a question is
        generated first. The question together with the fact text is then sent
        again to obtain a factual answer. Only a small portion of the graph is
        ever passed to the LLM which avoids context explosion.
        """

        fact_nodes = self._select_facts(kg, num_pairs)

        if not fact_nodes:
            return {"qa_pairs": []}

        temperature = self.generation_config.temperature
        max_tokens = self.generation_config.max_tokens

        question_prompt = get_prompt(self.config, "kg_question")
        answer_prompt = get_prompt(self.config, "kg_answer")

        qa_pairs: list[QAPair] = []

        if verbose:
            logger.info("Generating QA from %d facts", len(fact_nodes))

        for fid in fact_nodes:
            if len(qa_pairs) >= num_pairs:
                break

            fact = kg.graph.nodes[fid]
            fact_text = f"{fact['subject']} {fact['predicate']} {fact['object']}"

            context_chunks = [
                kg.graph.nodes[cid].get("text")
                for cid in kg.get_chunks_for_fact(fid)
                if "text" in kg.graph.nodes[cid]
            ]
            if context_chunks:
                fact_text += "\n" + "\n".join(context_chunks[:3])

            q_msg = question_prompt.format(facts=fact_text)
            question = self.client.chat_completion(
                [{"role": "system", "content": q_msg}],
                temperature=temperature,
                max_tokens=max_tokens,
            ).strip()

            a_msg = answer_prompt.format(facts=fact_text, question=question)
            answer = self.client.chat_completion(
                [{"role": "system", "content": a_msg}],
                temperature=temperature,
                max_tokens=max_tokens,
            ).strip()

            confidence = kg.fact_confidence(
                fact["subject"], fact["predicate"], fact["object"]
            )
            answers = 2 if multi_answer else 1
            for _ in range(answers):
                if len(qa_pairs) >= num_pairs:
                    break
                if _ > 0:
                    a_msg = answer_prompt.format(facts=fact_text, question=question)
                    answer = self.client.chat_completion(
                        [{"role": "system", "content": a_msg}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ).strip()
                qa_pairs.append(
                    QAPair(
                        question=question,
                        answer=answer,
                        facts=[fid],
                        confidence=confidence,
                    )
                )

        return {"qa_pairs": [p.to_dict() for p in qa_pairs]}
