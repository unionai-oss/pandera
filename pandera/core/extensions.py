"""Extensions module."""

from functools import partial, wraps
from inspect import signature, Parameter, Signature, _empty
from typing import Any, Callable, Dict, List, Optional, Type, Union

from multimethod import multidispatch
from pandera.core.checks import Check
from pandera.core.hypotheses import Hypothesis


# pylint: disable=too-many-locals
def register_check(
    fn=None,
    pre_init_hook: Optional[Callable] = None,
    aliases: Optional[List[str]] = None,
    strategy: Optional[Callable] = None,
    error: Optional[Union[str, Callable]] = None,
    check_cls: Type = Check,
    samples_kwtypes: Optional[Dict[str, Type]] = None,
    **outer_kwargs,
):
    """Register a check method to the Check namespace.

    This is the primary way for extending the Check api to define additional
    built-in checks.
    """

    if fn is None:
        return partial(
            register_check,
            pre_init_hook=pre_init_hook,
            aliases=aliases,
            strategy=strategy,
            error=error,
            check_cls=check_cls,
            samples_kwtypes=samples_kwtypes,
            **outer_kwargs,
        )

    name = fn.__name__

    # see if the check function is already registered
    check_fn = check_cls.CHECK_FUNCTION_REGISTRY.get(name)

    fn_sig = signature(fn)

    # this is a special case for handling hypotheses, since the sample keys
    # need to be treated like statistics and is used during preprocessing, not
    # in the check function itself.
    if samples_kwtypes is None:
        samples_args = []
        samples_params = []
    else:
        samples_args = [*samples_kwtypes]
        samples_params = [
            Parameter(
                name,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=samples_kwtypes[name],
            )
            for name in samples_kwtypes
        ]

    # derive statistics from function arguments after the 0th positional arg
    statistics = [*samples_args, *[*fn_sig.parameters.keys()][1:]]
    statistics_params = [
        *samples_params,
        *[*fn_sig.parameters.values()][1:],
    ]
    statistics_defaults = {
        p.name: p.default
        for p in fn_sig.parameters.values()
        if p.default is not _empty
    }

    if check_fn is None:

        dispatch_check_fn = multidispatch(fn)

        # create proxy function so we can modify the signature and docstring
        # of the method to reflect correctly in the documentation
        # pylint: disable=unused-argument
        def check_function_proxy(cls, *args, **kws):
            return dispatch_check_fn(*args, **kws)

        update_check_fn_proxy(
            check_cls, check_function_proxy, fn, fn_sig, statistics_params
        )

        @wraps(check_function_proxy)
        def check_method(cls, *args, **check_kwargs):
            args = list(args)

            statistics_kwargs = dict(zip(statistics, args))
            for stat in statistics:
                if stat in check_kwargs:
                    statistics_kwargs[stat] = check_kwargs.pop(stat)
                elif stat not in statistics_kwargs:
                    statistics_kwargs[stat] = statistics_defaults.get(
                        stat, None
                    )

            _check_kwargs = {
                "error": (
                    error.format(**statistics_kwargs)
                    if isinstance(error, str)
                    else error(**statistics_kwargs)
                )
            }
            _check_kwargs.update(outer_kwargs)
            _check_kwargs.update(check_kwargs)

            # this is a special case for handling hypotheses, since the sample
            # keys need to be treated like statistics and is used during
            # preprocessing, not in the check function itself.
            if samples_kwtypes is not None:
                samples = []
                for sample_arg in samples_kwtypes:
                    samples.append(statistics_kwargs.pop(sample_arg))
                _check_kwargs["samples"] = samples

            # This is kind of ugly... basically we're creating another
            # stats kwargs variable that's actually used when invoking the check
            # function (which may or may not be processed by pre_init_hook)
            # This is so that the original value is not modified by
            # pre_init_hook when, for e.g. the check is serialized with the io
            # module. Figure out a better way to do this!
            check_fn_stats_kwargs = (
                pre_init_hook(statistics_kwargs)
                if pre_init_hook is not None
                else statistics_kwargs
            )

            # internal wrapper is needed here to make sure the inner check_fn
            # produced by this method is consistent with the registered check
            # function
            if check_cls is Check:

                @wraps(fn)
                def _check_fn(check_obj, **inner_kwargs):
                    """
                    inner_kwargs will be based in via Check.__init__ kwargs.
                    """
                    # Raise more informative error when this fails to dispatch
                    return check_function_proxy(
                        cls,
                        check_obj,
                        *check_fn_stats_kwargs.values(),
                        **inner_kwargs,
                    )

            elif check_cls is Hypothesis:

                @wraps(fn)
                def _check_fn(*samples, **inner_kwargs):
                    """
                    inner_kwargs will be based in via Check.__init__ kwargs.
                    """
                    # Raise more informative error when this fails to dispatch
                    return check_function_proxy(
                        cls,
                        *samples,
                        **{
                            **check_fn_stats_kwargs,
                            **inner_kwargs,
                        },
                    )

            else:
                raise TypeError(f"check_cls {check_cls} not recognized")

            return cls(
                _check_fn,
                statistics=statistics_kwargs,
                strategy=(
                    None
                    if strategy is None
                    else partial(strategy, **statistics_kwargs)
                ),
                **_check_kwargs,
            )

        check_cls.CHECK_FUNCTION_REGISTRY[name] = dispatch_check_fn
        setattr(check_cls, name, classmethod(check_method))

        class_check_method = getattr(check_cls, name)

        for _name in [] if aliases is None else aliases:
            setattr(check_cls, _name, class_check_method)
    else:
        check_fn.register(fn)  # type: ignore

    return getattr(check_cls, name)


def register_hypothesis(samples_kwtypes=None, **kwargs):
    """Register a new hypothesis."""
    return partial(
        register_check,
        check_cls=Hypothesis,
        samples_kwtypes=samples_kwtypes,
        **kwargs,
    )


def generate_check_signature(
    check_cls: Type,
    sig: Signature,
    statistics_params: List[Parameter],
) -> Signature:
    """Generates a check signature from check statistics."""
    # assume the first argument is the check object
    return sig.replace(
        parameters=[
            # This first parameter will be ignored since it's the check object
            Parameter("_", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("cls", Parameter.POSITIONAL_OR_KEYWORD),
            *statistics_params,
            Parameter(
                "kwargs", Parameter.VAR_KEYWORD, annotation=Dict[str, Any]
            ),
        ],
        return_annotation=check_cls,
    )


def generate_check_annotations(
    check_cls: Type,
    statistics_params: List[Parameter],
) -> Dict[str, Type]:
    """Generates a check type annotations from check statistics."""
    return {
        **{p.name: p.annotation for p in statistics_params},
        "kwargs": Dict[
            str,
            Any,
        ],
        "return": check_cls,
    }


def modify_check_fn_doc(doc: str) -> str:
    """Adds"""
    return (
        f"{doc}\n{' ' * 4}:param kwargs: arguments forwarded to the "
        ":py:class:`~pandera.core.checks.Check` constructor."
    )


def update_check_fn_proxy(
    check_cls: Type, check_function_proxy, fn, fn_sig, statistics_params
):
    """
    Manually update the signature of `check_function` so that docstring matches
    original function's signature, but includes **kwargs, etc.
    """
    check_function_proxy.__name__ = fn.__name__
    check_function_proxy.__module__ = fn.__module__
    check_function_proxy.__qualname__ = fn.__qualname__
    check_function_proxy.__signature__ = generate_check_signature(
        check_cls,
        fn_sig,
        statistics_params,
    )
    check_function_proxy.__doc__ = modify_check_fn_doc(fn.__doc__)
    check_function_proxy.__annotations__ = generate_check_annotations(
        check_cls, statistics_params
    )
