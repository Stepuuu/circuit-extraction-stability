#!/usr/bin/env python3
"""
Logic-task generator with truly random variable names.

Design rationale:
- Variable names are fully random, drawn from no predefined vocabulary.
- The model must learn to
  1. recognize the variable names in the goal (whatever they are),
  2. find the matching hypothesis names among the premises,
  3. combine them according to the logical rules,
  so it learns pure logic rather than lexical preferences.

Variable-name generation:
1. random letter sequences: x, ab, xyz, qwerty
2. random letters + digits: a1, x2y, p3q4
3. single letters: p, q, r, s, ...
4. random length (1-6 characters)

Hypothesis-name generation:
1. h + digit: h0, h1, h2, ...
2. short random names: m, n, k, j, ...

This design ensures that variable names never seen during training are handled
correctly at test time: the model has to learn position and structure instead
of specific tokens.
"""

import random
import string
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# Truly Random Name Generator
# ============================================================

class TrulyRandomNameGenerator:
    """Generate fully random variable and hypothesis names."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.used_names = set()

    def reset(self):
        """Reset the used-name set (called at the start of every sample)."""
        self.used_names = set()

    def random_var_name(self, min_len: int = 1, max_len: int = 5) -> str:
        """
        Generate a random variable name.

        Strategies (uniform):
        1. pure lowercase letters: a, xy, abc, qwer
        2. letters + digit suffix: a1, xy2, abc3
        3. single letters
        """
        for _ in range(100):  # guard against infinite loops
            strategy = self.rng.choice(['alpha', 'alphanum', 'single'])

            if strategy == 'single':
                # Single letter
                name = self.rng.choice(string.ascii_lowercase)
            elif strategy == 'alpha':
                # Pure letters
                length = self.rng.randint(min_len, max_len)
                name = ''.join(self.rng.choices(string.ascii_lowercase, k=length))
            else:  # alphanum
                # Letters + digits
                alpha_len = self.rng.randint(1, max_len - 1)
                num_len = self.rng.randint(1, 2)
                alpha_part = ''.join(self.rng.choices(string.ascii_lowercase, k=alpha_len))
                num_part = ''.join(self.rng.choices(string.digits, k=num_len))
                name = alpha_part + num_part

            # Ensure uniqueness
            if name not in self.used_names:
                self.used_names.add(name)
                return name

        # Fallback: append a random suffix
        name = ''.join(self.rng.choices(string.ascii_lowercase, k=3)) + str(self.rng.randint(100, 999))
        self.used_names.add(name)
        return name

    def random_hyp_name(self) -> str:
        """
        Generate a random hypothesis name.

        Strategies:
        1. h + digit: h0, h1, h2, ...
        2. single letters: m, n, k, j
        3. two letters: ha, hb, ...
        """
        for _ in range(100):
            strategy = self.rng.choice(['h_num', 'single', 'double'])

            if strategy == 'h_num':
                name = 'h' + str(self.rng.randint(0, 99))
            elif strategy == 'single':
                # Avoid the common proposition letters p, q, r
                name = self.rng.choice(['m', 'n', 'k', 'j', 'f', 'g', 'w'])
            else:  # double
                name = 'h' + self.rng.choice(string.ascii_lowercase)

            if name not in self.used_names:
                self.used_names.add(name)
                return name

        # Fallback
        name = 'h' + str(self.rng.randint(100, 999))
        self.used_names.add(name)
        return name


# ============================================================
# Sample Data Structure
# ============================================================

@dataclass
class LogicSample:
    """One logic proof sample."""
    task_type: str              # S_and, T_and_or, etc.
    props: List[str]  # proposition names
    hyps: List[str]  # hypothesis names
    premises: List[Tuple[str, str]]  # (hypothesis, proposition) pairs
    distractors: List[Tuple[str, str]]  # distractor premises
    goal: str  # goal
    tactic: str  # proof tactic

    def to_input_output(self) -> Tuple[str, str]:
        """Convert to the input/output format."""
        # Build the premises section (real premises mixed with distractors)
        all_premises = self.premises + self.distractors
        random.shuffle(all_premises)

        # Proposition declarations
        all_props = list(set(self.props + [p for _, p in self.distractors]))
        prop_decl = ' '.join(all_props) + ' : Prop'

        # Premise declarations
        premise_lines = [f"{h} : {p}" for h, p in all_premises]

        # Build the input
        input_text = f"state_0:\n{prop_decl}\n" + '\n'.join(premise_lines) + f"\n⊢ {self.goal}"

        # Build the output
        output_text = f"state_0_tactic_0:\n{self.tactic}\nstate_1:\nno goals\nproof complete"

        return input_text, output_text


# ============================================================
# Logic task generators (truly random names)
# ============================================================

class TrulyRandomDatasetGenerator:
    """
    Generate logic proof datasets with fully random variable names.

    Core principles:
    1. variable names are fully random, drawn from no vocabulary
    2. the model must learn pure logic, not lexical patterns
    3. distractor premises increase difficulty
    """

    def __init__(self, seed: int = 42, n_distractors: int = 2):
        self.seed = seed
        self.name_gen = TrulyRandomNameGenerator(seed)
        self.n_distractors = n_distractors
        self.rng = random.Random(seed)

    def _get_random_props(self, n: int) -> List[str]:
        """Get n random proposition names."""
        return [self.name_gen.random_var_name() for _ in range(n)]

    def _get_random_hyps(self, n: int) -> List[str]:
        """Get n random hypothesis names."""
        return [self.name_gen.random_hyp_name() for _ in range(n)]

    def _get_distractors(self, n: int) -> List[Tuple[str, str]]:
        """Generate distractor premises."""
        distractors = []
        for _ in range(n):
            h = self.name_gen.random_hyp_name()
            p = self.name_gen.random_var_name()
            distractors.append((h, p))
        return distractors

    # ========== Atomic Skills ==========

    def generate_S_and(self) -> LogicSample:
        """
        S_and: h1:p, h2:q ⊢ p ∧ q
        Tactic: exact And.intro h1 h2
        """
        self.name_gen.reset()

        p, q = self._get_random_props(2)
        h1, h2 = self._get_random_hyps(2)

        premises = [(h1, p), (h2, q)]
        goal = f"{p} ∧ {q}"
        tactic = f"exact And.intro {h1} {h2}"

        return LogicSample(
            task_type='S_and',
            props=[p, q],
            hyps=[h1, h2],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_S_or_l(self) -> LogicSample:
        """
        S_or_l: h:p ⊢ p ∨ q
        Tactic: exact Or.inl h

        Note: q appears in the goal without a matching hypothesis (the model must infer Or.inl).
        """
        self.name_gen.reset()

        p, q = self._get_random_props(2)
        h = self._get_random_hyps(1)[0]

        premises = [(h, p)]  # provide p only, not q
        goal = f"{p} ∨ {q}"
        tactic = f"exact Or.inl {h}"

        return LogicSample(
            task_type='S_or_l',
            props=[p, q],
            hyps=[h],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_S_or_r(self) -> LogicSample:
        """
        S_or_r: h:q ⊢ p ∨ q
        Tactic: exact Or.inr h
        """
        self.name_gen.reset()

        p, q = self._get_random_props(2)
        h = self._get_random_hyps(1)[0]

        premises = [(h, q)]  # provide q only, not p
        goal = f"{p} ∨ {q}"
        tactic = f"exact Or.inr {h}"

        return LogicSample(
            task_type='S_or_r',
            props=[p, q],
            hyps=[h],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_S_id(self) -> LogicSample:
        """
        S_id: h:p ⊢ p
        Tactic: exact h
        """
        self.name_gen.reset()

        p = self._get_random_props(1)[0]
        h = self._get_random_hyps(1)[0]

        premises = [(h, p)]
        goal = p
        tactic = f"exact {h}"

        return LogicSample(
            task_type='S_id',
            props=[p],
            hyps=[h],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    # ========== Composite Tasks ==========

    def generate_T_and_or(self) -> LogicSample:
        """
        T_and_or: h1:p, h2:q ⊢ (p ∧ q) ∨ r
        Tactic: exact Or.inl (And.intro h1 h2)

        Variant: the Or.inr path.
        """
        self.name_gen.reset()

        p, q, r = self._get_random_props(3)
        h1, h2 = self._get_random_hyps(2)

        # Randomly choose the left or right path
        if self.rng.choice([True, False]):
            # Left path: provide p, q
            premises = [(h1, p), (h2, q)]
            goal = f"({p} ∧ {q}) ∨ {r}"
            tactic = f"exact Or.inl (And.intro {h1} {h2})"
        else:
            # Right path: provide r
            h = h1
            premises = [(h, r)]
            goal = f"({p} ∧ {q}) ∨ {r}"
            tactic = f"exact Or.inr {h}"

        return LogicSample(
            task_type='T_and_or',
            props=[p, q, r],
            hyps=[h1, h2] if len(premises) > 1 else [h1],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_T_or_and(self) -> LogicSample:
        """
        T_or_and: ⊢ (p ∨ q) ∧ r
        Variant: the Or.inl or Or.inr path.
        """
        self.name_gen.reset()

        p, q, r = self._get_random_props(3)
        h1, h2 = self._get_random_hyps(2)

        if self.rng.choice([True, False]):
            # Inner left: provide p, r
            premises = [(h1, p), (h2, r)]
            goal = f"({p} ∨ {q}) ∧ {r}"
            tactic = f"exact And.intro (Or.inl {h1}) {h2}"
        else:
            # Inner right: provide q, r
            premises = [(h1, q), (h2, r)]
            goal = f"({p} ∨ {q}) ∧ {r}"
            tactic = f"exact And.intro (Or.inr {h1}) {h2}"

        return LogicSample(
            task_type='T_or_and',
            props=[p, q, r],
            hyps=[h1, h2],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_T_nested_and(self) -> LogicSample:
        """
        T_nested_and: h1:p, h2:q, h3:r ⊢ (p ∧ q) ∧ r
        Tactic: exact And.intro (And.intro h1 h2) h3
        """
        self.name_gen.reset()

        p, q, r = self._get_random_props(3)
        h1, h2, h3 = self._get_random_hyps(3)

        premises = [(h1, p), (h2, q), (h3, r)]
        goal = f"({p} ∧ {q}) ∧ {r}"
        tactic = f"exact And.intro (And.intro {h1} {h2}) {h3}"

        return LogicSample(
            task_type='T_nested_and',
            props=[p, q, r],
            hyps=[h1, h2, h3],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_T_nested_or(self) -> LogicSample:
        """
        T_nested_or: ⊢ (p ∨ q) ∨ r
        Variant: three possible paths.
        """
        self.name_gen.reset()

        p, q, r = self._get_random_props(3)
        h = self._get_random_hyps(1)[0]

        choice = self.rng.choice(['p', 'q', 'r'])
        goal = f"({p} ∨ {q}) ∨ {r}"

        if choice == 'p':
            premises = [(h, p)]
            tactic = f"exact Or.inl (Or.inl {h})"
        elif choice == 'q':
            premises = [(h, q)]
            tactic = f"exact Or.inl (Or.inr {h})"
        else:  # r
            premises = [(h, r)]
            tactic = f"exact Or.inr {h}"

        return LogicSample(
            task_type='T_nested_or',
            props=[p, q, r],
            hyps=[h],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    def generate_T_and_or_and(self) -> LogicSample:
        """
        T_and_or_and: ⊢ (p ∧ q) ∨ (r ∧ s)
        Variant: left or right path.
        """
        self.name_gen.reset()

        p, q, r, s = self._get_random_props(4)
        h1, h2 = self._get_random_hyps(2)

        goal = f"({p} ∧ {q}) ∨ ({r} ∧ {s})"

        if self.rng.choice([True, False]):
            # Left: provide p, q
            premises = [(h1, p), (h2, q)]
            tactic = f"exact Or.inl (And.intro {h1} {h2})"
        else:
            # Right: provide r, s
            premises = [(h1, r), (h2, s)]
            tactic = f"exact Or.inr (And.intro {h1} {h2})"

        return LogicSample(
            task_type='T_and_or_and',
            props=[p, q, r, s],
            hyps=[h1, h2],
            premises=premises,
            distractors=self._get_distractors(self.n_distractors),
            goal=goal,
            tactic=tactic,
        )

    # ========== Batch Generation ==========

    def generate_sample(self, task_type: str = None) -> LogicSample:
        """Generate a single sample."""
        generators = {
            'S_and': self.generate_S_and,
            'S_or_l': self.generate_S_or_l,
            'S_or_r': self.generate_S_or_r,
            'S_id': self.generate_S_id,
            'T_and_or': self.generate_T_and_or,
            'T_or_and': self.generate_T_or_and,
            'T_nested_and': self.generate_T_nested_and,
            'T_nested_or': self.generate_T_nested_or,
            'T_and_or_and': self.generate_T_and_or_and,
        }

        if task_type is None:
            task_type = self.rng.choice(list(generators.keys()))

        return generators[task_type]()


# ============================================================
# Demo
# ============================================================

def demo():
    """Demo of truly random data generation."""
    print("=" * 70)
    print("Truly Random Logic Dataset Generator Demo")
    print("=" * 70)
    print()

    gen = TrulyRandomDatasetGenerator(seed=12345, n_distractors=2)

    for task in ['S_and', 'S_or_l', 'T_and_or', 'T_nested_and']:
        sample = gen.generate_sample(task)
        input_text, output_text = sample.to_input_output()

        print(f"[{task}]")
        print(f"Props: {sample.props}")
        print(f"Hyps: {sample.hyps}")
        print(f"Input:\n{input_text}")
        print(f"Output:\n{output_text}")
        print("-" * 70)
        print()


if __name__ == "__main__":
    demo()
