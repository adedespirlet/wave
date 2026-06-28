# Copyright 2025, The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, TypeAlias

import sympy  # type: ignore

# ============================================================================
# Monkey-patch for SymPy bug #28744 (affects SymPy 1.13.3).
#
# In sympy/core/mod.py Mod.eval(), when p is a Mul containing a Mod factor,
# the code rebuilds `non_mod_l` by iterating p.args instead of the already-
# separated non_mod_l list.  That makes the Mod element appear in BOTH
# non_mod_l and mod_l, so the final Mul(*non_mod_l + mod_l) squares it:
#   Mod(4 * Mod(T0, 16), 32) → 4 * Mod(T0, 16)**2  (wrong)
#
# The fix: iterate over non_mod_l (the non-Mod factors) instead of p.args.
# This is safe because Mod factors are already captured in mod_l and must
# not be included again.
# ============================================================================
def _patched_mod_eval(cls, p, q):
    from sympy.core.mod import Mod
    from sympy.core.mul import Mul
    from sympy.core.add import Add
    from sympy.core.singleton import S
    from sympy.core.numbers import equal_valued
    from sympy.core.exprtools import gcd_terms

    def number_eval(p, q):
        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")
        if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
            return S.NaN
        if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
            return S.Zero
        if q.is_Number:
            if p.is_Number:
                return p % q
            if q == 2:
                if p.is_even:
                    return S.Zero
                elif p.is_odd:
                    return S.One
        if hasattr(p, "_eval_Mod"):
            rv = getattr(p, "_eval_Mod")(q)
            if rv is not None:
                return rv
        # by ratio
        r = p / q
        if r.is_integer:
            return S.Zero
        try:
            d = int(r)
        except TypeError:
            pass
        else:
            if isinstance(d, int):
                rv = p - d * q
                if (rv * q < 0) == True:
                    rv += q
                return rv
        # by difference: -2|q| < p < 2|q|
        d = abs(p)
        for _ in range(2):
            d -= abs(q)
            if d.is_negative:
                if q.is_positive:
                    if p.is_positive:
                        return d + q
                    elif p.is_negative:
                        return -d
                elif q.is_negative:
                    if p.is_positive:
                        return d
                    elif p.is_negative:
                        return -d + q
                break

    rv = number_eval(p, q)
    if rv is not None:
        return rv

    # denest
    if isinstance(p, cls):
        qinner = p.args[1]
        if qinner % q == 0:
            return cls(p.args[0], q)
        elif (qinner * (q - qinner)).is_nonnegative:
            return p
    elif isinstance(-p, cls):
        qinner = (-p).args[1]
        if qinner % q == 0:
            return cls(-(-p).args[0], q)
        elif (qinner * (q + qinner)).is_nonpositive:
            return p
    elif isinstance(p, Add):
        both_l = non_mod_l, mod_l = [], []
        for arg in p.args:
            both_l[isinstance(arg, cls)].append(arg)
        if mod_l and all(inner.args[1] == q for inner in mod_l):
            net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
            return cls(net, q)

    elif isinstance(p, Mul):
        both_l = non_mod_l, mod_l = [], []
        for arg in p.args:
            both_l[isinstance(arg, cls)].append(arg)

        if mod_l and all(inner.args[1] == q for inner in mod_l) and all(t.is_integer for t in p.args) and q.is_integer:
            non_mod_l = [cls(x, q) for x in non_mod_l]
            mod = []
            non_mod = []
            for j in non_mod_l:
                if isinstance(j, cls):
                    mod.append(j.args[0])
                else:
                    non_mod.append(j)
            prod_mod = Mul(*mod)
            prod_non_mod = Mul(*non_mod)
            prod_mod1 = Mul(*[i.args[0] for i in mod_l])
            net = prod_mod1 * prod_mod
            return prod_non_mod * cls(net, q)

        if q.is_Integer and q is not S.One:
            if all(t.is_integer for t in p.args):
                # FIX for SymPy bug #28744: iterate non_mod_l, not p.args.
                # Using p.args would include Mod elements that are already in
                # mod_l, causing them to be doubled when building Mul below.
                non_mod_l = [i % q if i.is_Integer else i for i in non_mod_l]
                if any(iq is S.Zero for iq in non_mod_l):
                    return S.Zero

        p = Mul(*(non_mod_l + mod_l))

    from sympy.polys.polyerrors import PolynomialError
    from sympy.polys.polytools import gcd

    try:
        G = gcd(p, q)
        if not equal_valued(G, 1):
            p, q = [gcd_terms(i / G, clear=False, fraction=False) for i in (p, q)]
    except PolynomialError:
        G = S.One
    pwas, qwas = p, q

    if p.is_Add:
        args = []
        for i in p.args:
            a = cls(i, q)
            if a.count(cls) > i.count(cls):
                args.append(i)
            else:
                args.append(a)
        if args != list(p.args):
            p = Add(*args)
    else:
        cp, p = p.as_coeff_Mul()
        cq, q = q.as_coeff_Mul()
        ok = False
        if not cp.is_Rational or not cq.is_Rational:
            r = cp % cq
            if equal_valued(r, 0):
                G *= cq
                p *= int(cp / cq)
                ok = True
        if not ok:
            p = cp * p
            q = cq * q

    if p.could_extract_minus_sign() and q.could_extract_minus_sign():
        G, p, q = [-i for i in (G, p, q)]

    rv = number_eval(p, q)
    if rv is not None:
        return rv * G

    if G.is_Float and equal_valued(G, 1):
        p *= G
        return cls(p, q, evaluate=False)
    elif G.is_Mul and G.args[0].is_Float and equal_valued(G.args[0], 1):
        p = G.args[0] * p
        G = Mul._from_args(G.args[1:])
    return G * cls(p, q, evaluate=(p, q) != (pwas, qwas))


sympy.Mod.eval = classmethod(_patched_mod_eval)

__all__ = [
    "sym",
    "IndexExpr",
    "IndexSequence",
    "IndexSymbol",
    "index_symbol",
    "index_expr",
    "piecewise_aware_subs",
    "MMA_ACC_SYMBOL_NAME",
    "THREAD_SYMBOL_NAMES",
    "WORKGROUP_SYMBOL_NAMES",
    "DEVICE_SYMBOL_NAMES",
    "GPR_SYMBOL_NAME",
]

MMA_ACC_SYMBOL_NAME = "$MMA_ACC"
THREAD_SYMBOL_NAMES = ("$T0", "$T1", "$T2")
WORKGROUP_SYMBOL_NAMES = ("$WG0", "$WG1", "$WG2")
DEVICE_SYMBOL_NAMES = ("$DD0", "$DD1", "$DD2")
GPR_SYMBOL_NAME = "$GPR_NUM"

###############################################################################
# Index symbols and expressions
# These are just light-weight helpers around sympy symbols and expressions.
###############################################################################

IndexSymbol: TypeAlias = sympy.Symbol
IndexExpr: TypeAlias = sympy.Expr


def index_symbol(name: str) -> IndexSymbol:
    """Returns a named symbol, assumed to be a non-negative integer."""
    return sympy.Symbol(name, integer=True, nonnegative=True)


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n) -> IndexSymbol:
        return index_symbol(n)


sym = _IndexSymbolExpando()


def _subs_piecewise_walk(expr, subs_dict):
    if isinstance(expr, sympy.Piecewise):
        return sympy.Piecewise(
            *[(e.subs(subs_dict), c.subs(subs_dict)) for e, c in expr.args]
        )

    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        if isinstance(expr, sympy.Symbol) and expr in subs_dict:
            return subs_dict[expr]
        return expr

    if not expr.has(sympy.Piecewise):
        return expr.subs(subs_dict)

    new_args = [_subs_piecewise_walk(arg, subs_dict) for arg in expr.args]
    return expr.func(*new_args)


def piecewise_aware_subs(
    expr: "IndexExpr",
    subs_dict,
    simultaneous: bool = False,
) -> "IndexExpr":
    """Substitute into expr, handling Piecewise nodes efficiently.

    Avoids sympy's expensive recursive boolean simplification in
    Piecewise._eval_subs by substituting into each (value, condition) pair of a
    Piecewise independently.

    For expressions without Piecewise, delegates to sympy's normal subs().

    subs_dict can be a dict or a list of (old, new) pairs, matching sympy convention.
    """
    if not isinstance(expr, sympy.Basic):
        return expr

    if simultaneous:
        return expr.subs(subs_dict, simultaneous=True)

    if isinstance(subs_dict, (list, tuple)):
        subs_dict = dict(subs_dict)

    expr_syms = expr.free_symbols
    subs_keys = set(subs_dict.keys())
    matching = expr_syms & subs_keys
    if not matching:
        return expr

    filtered = {k: v for k, v in subs_dict.items() if k in matching}

    if isinstance(expr, sympy.Piecewise):
        return sympy.Piecewise(
            *[(e.subs(filtered), c.subs(filtered)) for e, c in expr.args]
        )
    elif expr.has(sympy.Piecewise):
        return _subs_piecewise_walk(expr, filtered)
    else:
        return expr.subs(filtered)


@dataclass
class IndexSequence:
    start: IndexExpr | int
    size: IndexExpr | int
    stride: IndexExpr | int = 1

    @staticmethod
    def _subs(
        value: int | IndexExpr,
        map: dict[IndexExpr, IndexExpr],
        simultaneous: bool = False,
    ) -> int | IndexExpr:
        if isinstance(value, IndexSequence):
            return value.subs(map, simultaneous=simultaneous)
        if isinstance(value, sympy.Basic):
            return piecewise_aware_subs(value, map, simultaneous=simultaneous)
        return value

    def has(self, symbol: IndexSymbol) -> bool:
        return (
            sympy.sympify(self.start).has(symbol)
            or sympy.sympify(self.size).has(symbol)
            or sympy.sympify(self.stride).has(symbol)
        )

    def subs(self, map: dict[IndexExpr, IndexExpr], simultaneous: bool = False):
        start = self._subs(self.start, map, simultaneous)
        size = self._subs(self.size, map, simultaneous)
        stride = self._subs(self.stride, map, simultaneous)
        return IndexSequence(start, size, stride)

    @staticmethod
    def from_expr(expr: IndexExpr, subs: dict[IndexExpr, Any]):
        start_subs = {k: v.start for k, v in subs.items()}
        size_subs = {k: v.size for k, v in subs.items()}
        stride_subs = {k: v.stride for k, v in subs.items()}
        start = IndexSequence._subs(expr, start_subs)
        size = IndexSequence._subs(expr, size_subs)
        stride = IndexSequence._subs(expr, stride_subs)
        return IndexSequence(start, size, stride)

    def __repr__(self) -> str:
        return f"{self.start} : {self.size} : {self.stride}"

    def __hash__(self):
        return hash((self.start, self.size, self.stride))
