
function append_tooltip_radar_chart_and_heading(parent_div_id, col, deviation) {


    let tippy_div_width = 300;

    const col_name_font_size = 16;

    if (col[key_header].length*8 > tippy_div_width) {
        tippy_div_width = col[key_header].length*8;
    }

    let tippy_div;

    if (d3.select('#' + col[key_id] + id_tooltip_div_ending).empty()) {
        tippy_div = d3.select('#' + parent_div_id).append('div')
            .attr('id', col[key_id] + id_tooltip_div_ending)
            .style('width', tippy_div_width + 'px')
            .style('height', 280 + 'px')
            .style('float', 'left')
            .style('display', 'block')
            .style('display', 'none');
    } else {
        tippy_div = d3.select('#' + col[key_id] + id_tooltip_div_ending);
        tippy_div.selectAll('*').remove();
    }

    tippy_div.style('width', (tippy_div_width) + 'px');


    let radar_chart_div = tippy_div.append('div')
        .attr('id', col[key_id]  + id_radarChartDiv_class)
        .style('height', 280 - 25 +'px')
        .style('width', (tippy_div_width+0)+'px');


    append_radar_chart_tooltip(radar_chart_div, col[key_data_type], col, deviation);


    let text_svg = tippy_div.append('div')
        .style('width', tippy_div_width + 'px')
        .style('height', 25 + 'px')
        .append('svg')
        .style('width', '100%')
        .style('height', '100%')
    // text_svg.append('rect')
    //     .style('width', '100%')
    //     .style('height', '100%')
        //.style('fill', 'green');

    text_svg.append('text')
        .text(function () {
           return col[key_header];
        })
        .attr('fill', 'white')
        .attr('font-size', col_name_font_size +'px')
        .attr('font-weight', 500)
        .attr('text-anchor', 'middle')
        .attr('transform', 'translate(' + ((tippy_div_width) / 2) + ',' + 33 / 2 + ')');

    return tippy_div.node().innerHTML;
}



