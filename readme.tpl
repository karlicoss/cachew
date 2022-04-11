{# disable code used to generate readme #}
{# based on https://stackoverflow.com/a/55305881/706389 #}

{%- extends 'markdown/index.md.j2' -%}

{% block input_group %}
    {%- if cell.metadata.get('nbconvert', {}).get('show_code', False) -%}
        ((( super() )))
    {%- endif -%}
{% endblock input_group %}
