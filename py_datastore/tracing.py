"""
AioHttp Client tracing, only compatible with Aiohttp 3.X versions
"""
import aiohttp

from jaeger_client import Config
from namedlist import namedlist
from opentracing.ext import tags
from opentracing.propagation import Format

def init_tracer(service):
    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'logging': False,
            'reporter_batch_size': 1,
        },
        service_name=service
    )
    return config.initialize_tracer()

def instrument_sanic(app, tracer):
    @app.middleware('request')
    async def start_trace(request):
        route_name = request.app.router.get(request)[0].__name__
        request["span"] = tracer.start_span(route_name).__enter__()
        request["span"].set_tag(tags.HTTP_METHOD, request.method)
        request["span"].set_tag(tags.HTTP_URL, request.url)
        request["span"].set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_SERVER)

    @app.middleware('response')
    async def end_trace(request, response):
        if "span" in request:
            request["span"].set_tag(tags.HTTP_STATUS_CODE, response.status)
            request["span"].__exit__(None, None, None)

class AiohttpTracer(aiohttp.TraceConfig):
    def __init__(self, tracer):
        Ctx = namedlist('Ctx', ['parentspan', 'span'], default=None)
        async def on_request_start(session, ctx, params):
            ctx.span = tracer.start_span('aiohttp', child_of=ctx.parentspan).__enter__()
            ctx.span.set_tag(tags.HTTP_METHOD, params.method)
            ctx.span.set_tag(tags.HTTP_URL, params.url.human_repr())
            ctx.span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_CLIENT)

        async def on_request_end(session, ctx, params):
            ctx.span.set_tag(tags.HTTP_STATUS_CODE, params.response.status)
            if params.response.status != 200:
                ctx.span.__exit__(None, None, None)

        async def on_request_exception(session, ctx, params):
            ctx.span.set_tag(tags.ERROR, params.exception)
            ctx.span.__exit__(None, None, None)

        async def on_response_chunk_received(session, ctx, params):
            ctx.span.__exit__(None, None, None)

        def ctx_factory(trace_request_ctx):
            return Ctx(parentspan=trace_request_ctx)
        super().__init__(trace_config_ctx_factory=ctx_factory)
        self.on_request_start.append(on_request_start)
        self.on_request_end.append(on_request_end)
        self.on_request_exception.append(on_request_exception)
        self.on_response_chunk_received.append(on_response_chunk_received)
