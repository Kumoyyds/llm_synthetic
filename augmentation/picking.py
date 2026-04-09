"""
Synthetic data generation by picking similar/dissimilar concepts.
Extracts logic from similarity_check.ipynb into a reusable class.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Literal, Tuple
from pathlib import Path

from models import similar as sm
from models import need_filter as nf


class ConceptPicker:
    """
    Generate synthetic observations by finding similar or dissimilar concepts
    and filtering them through an AI model based on respondent profiles.
    """

    KPI_MAPPING = {
        "relevance": "RelFlag",
        "differentiation": "DiffFlag",
        "believability": "BelFlag"
    }

    def __init__(
        self,
        food_concepts: Dict,
        all_us_food_concepts: pd.DataFrame,
        transformed_respondent: Dict,
        response_table: pd.DataFrame,
        cache_path: str = "../data/embeddings_cache/us_food_concepts.pkl",
        # Similarity thresholds
        cate_match_bound: float = 0.5,
        contra_lower_bound: float = 0.28,
        same_lower_bound: float = 0.5,
        same_upper_bound: float = 0.75,
    ):
        """
        Initialize the ConceptPicker with data and parameters.

        Args:
            food_concepts: Dict mapping concept_id -> {concept_Cate, concept_content, ...}
            all_us_food_concepts: DataFrame with ConceptText, CAT2, RelFlag, DiffFlag, BelFlag
            transformed_respondent: Dict mapping respondent_id -> respondent profile info
            response_table: DataFrame with ids, question, concept, answer columns
            cache_path: Path for embedding cache
            cate_match_bound: Minimum similarity for category matching
            contra_lower_bound: Maximum similarity for contra (dissimilar) concepts
            same_lower_bound: Minimum similarity for same-type concepts
            same_upper_bound: Maximum similarity for same-type concepts
        """
        self.food_concepts = food_concepts
        self.all_us_food_concepts = all_us_food_concepts.copy()
        self.transformed_respondent = transformed_respondent
        self.response_table = response_table.copy()

        self.cache_path = cache_path
        self.cate_match_bound = cate_match_bound
        self.contra_lower_bound = contra_lower_bound
        self.same_lower_bound = same_lower_bound
        self.same_upper_bound = same_upper_bound

        # Deduplicate concepts
        self.all_us_food_concepts.drop_duplicates(subset=['ConceptText'], inplace=True)
        self.all_us_food_concepts.reset_index(drop=True, inplace=True)

        # Build category list and searcher
        self.us_food_cates = list(set(
            self.all_us_food_concepts.dropna(subset=['CAT2'])['CAT2']
        ))
        self.searcher_cate = sm.SimilaritySearcher()
        print("prepare the cate matcher")
        self.searcher_cate.fit(self.us_food_cates, cache_path=self.cache_path)


    def _determine_process_type(self, sub: pd.DataFrame) -> Literal['contra', 'same']:
        """Determine whether to find contra or same-type concepts."""
        if len(sub['answer'].value_counts()) == 1:
            return 'contra'
        return 'same'

    def _get_filtered_concepts(
        self,
        kpi: str,
        answer: str,
        process_type: str,
        suitable_cates: Optional[set] = None
    ) -> pd.DataFrame:
        """Get concepts filtered by KPI flag and optionally category."""
        kpi_col = self.KPI_MAPPING[kpi]

        if process_type == 'contra':
            if answer == 'yes':
                kpi_mask = self.all_us_food_concepts[kpi_col].isin(['L', 'ML'])
            else:
                kpi_mask = self.all_us_food_concepts[kpi_col].isin(['H', 'MH'])

            if suitable_cates:
                out_mask = kpi_mask & self.all_us_food_concepts['CAT2'].isin(suitable_cates)
            else:
                out_mask = kpi_mask
        else:  # same
            if answer == 'yes':
                out_mask = self.all_us_food_concepts[kpi_col].isin(['H', 'MH'])
            else:
                out_mask = self.all_us_food_concepts[kpi_col].isin(['L', 'ML'])

        return self.all_us_food_concepts[out_mask]

    def _find_candidates(
        self,
        query_content: str,
        filtered_concepts: pd.DataFrame,
        process_type: str
    ) -> List[str]:
        """Find candidate concepts based on similarity."""
        corpus = list(filtered_concepts['ConceptText'])
        if not corpus:
            return []

        searcher_concept = sm.SimilaritySearcher()
        searcher_concept.fit(corpus, cache_path=self.cache_path)

        top_results, bottom_results = searcher_concept.search(
            query_content, top_n=3, bottom_m=3
        )

        if process_type == 'contra':
            candidates = [
                text for text, score in bottom_results
                if score <= self.contra_lower_bound
            ]
        else:  # same
            candidates = [
                text for text, score in top_results
                if self.same_lower_bound <= score <= self.same_upper_bound
            ]

        return candidates

    async def process_respondent_kpi(
        self,
        respondent_id: str,
        kpi: str,
        max_concurrency: int = 6,
        show_progress: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process a single respondent + KPI combination.

        Args:
            respondent_id: ID of the respondent
            kpi: One of "relevance", "differentiation", "believability"
            max_concurrency: Max concurrent AI filter requests
            show_progress: Show tqdm progress bar

        Returns:
            Tuple of (real_df, synthetic_df) DataFrames
        """
        # Get real responses
        mask = (self.response_table['question'] == kpi) & (self.response_table['ids'] == respondent_id)
        sub = self.response_table[mask].copy()

        if sub.empty:
            return pd.DataFrame(), pd.DataFrame()

        sub['ob_type'] = 'real'
        sub['corresponding_concept'] = sub['concept']
        real_df = sub.copy()

        # Determine process type
        process_type = self._determine_process_type(sub)

        # Collect synthetic results
        final_result = {
            'ids': [],
            'concept': [],
            'question': [],
            'answer': [],
            'ob_type': [],
            'corresponding_concept': []
        }

        respondent_info = self.transformed_respondent[respondent_id]

        for i in range(len(sub)):
            item = sub.iloc[i]
            concept_id = str(item['concept'])
            original_answer = item['answer']

            # Get query info
            query_cate = self.food_concepts[concept_id]['concept_Cate']
            query_content = self.food_concepts[concept_id]['concept_content']

            # Find suitable categories for contra
            suitable_cates = None
            if process_type == 'contra':
                top_results, _ = self.searcher_cate.search(query_cate, top_n=5, bottom_m=0)
                suitable_cates = {c[0] for c in top_results if c[1] >= self.cate_match_bound}

            # Get filtered concepts
            filtered_df = self._get_filtered_concepts(
                kpi, original_answer, process_type, suitable_cates
            )

            if filtered_df.empty:
                continue

            # Find candidates
            candidates = self._find_candidates(query_content, filtered_df, process_type)
            if not candidates:
                continue

            # Build batch items
            items = [
                {
                    "new_concept": concept,
                    "kpi_type": kpi,
                    "system_info": respondent_info,
                    "return_reasoning": False
                }
                for concept in candidates
            ]

            # Run AI filter
            batch_results = await nf.ai_filter_batch_async(
                items,
                max_concurrency=max_concurrency,
                show_progress=show_progress,
                progress_desc=f"AI filter ({process_type})"
            )

            # Determine expected answer
            if process_type == 'contra':
                pick_answer = 'no' if original_answer == 'yes' else 'yes'
                ob_type = 'synthetic_contra'
            else:
                pick_answer = original_answer
                ob_type = 'synthetic_same'

            # Collect matching results
            for k, result in enumerate(batch_results):
                if result == pick_answer:
                    final_result['ids'].append(respondent_id)
                    final_result['concept'].append(candidates[k])
                    final_result['question'].append(kpi)
                    final_result['answer'].append(result)
                    final_result['ob_type'].append(ob_type)
                    final_result['corresponding_concept'].append(concept_id)

        synthetic_df = pd.DataFrame(final_result)
        return real_df, synthetic_df

    async def process_all(
        self,
        respondent_ids: Optional[List[str]] = None,
        kpis: Optional[List[str]] = None,
        max_concurrency: int = 4,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process all respondent + KPI combinations.

        Args:
            respondent_ids: List of respondent IDs to process (default: all)
            kpis: List of KPIs to process (default: all)
            max_concurrency: Max concurrent AI filter requests
            show_progress: Show progress bar

        Returns:
            Combined DataFrame with real and synthetic observations
        """
        if respondent_ids is None:
            respondent_ids = list(self.transformed_respondent.keys())

        if kpis is None:
            kpis = ["relevance", "differentiation", "believability"]

        all_results = []
        total = len(respondent_ids) * len(kpis)

        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total, desc="Processing respondents") if show_progress else None
        except ImportError:
            pbar = None

        for resp_id in respondent_ids:
            for kpi in kpis:
                real_df, synthetic_df = await self.process_respondent_kpi(
                    resp_id, kpi, max_concurrency=max_concurrency, show_progress=False
                )
                if not real_df.empty:
                    all_results.append(real_df)
                if not synthetic_df.empty:
                    all_results.append(synthetic_df)

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
