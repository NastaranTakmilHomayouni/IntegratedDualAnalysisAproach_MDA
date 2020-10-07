let overplotted_variables = [];

function overplotting_tooltip(d, xScale, yScale, x_axis_key, y_axis_key, this_, parent_div_id, possible_correlations_bool) {
    d3.select(this_).style('opacity', 1);

    d3.select('#overplotting_tooltip_div').remove();
    for (let index_dims = 0; index_dims < overplotted_variables.length; index_dims++) {
        d3.select('#' + overplotted_variables[index_dims][key_id] + id_circle_dimension_ending).style('stroke-width', 0 + 'px');
    }

    let current_descr_statistics_x = xScale(d.descriptive_statistics[x_axis_key]);
    let current_descr_statistics_y = yScale(d.descriptive_statistics[y_axis_key]);

    if (y_axis_key === key_data_type) {
        current_descr_statistics_y = yScale(d[y_axis_key]);
    }

    let filtered = column_values_filtered.filter(x => xScale(x.descriptive_statistics[x_axis_key]) < (current_descr_statistics_x + range_mouseover) && xScale(x.descriptive_statistics[x_axis_key]) > (current_descr_statistics_x - range_mouseover));
    let filtered_ = filtered.filter(y => yScale(y.descriptive_statistics[y_axis_key]) < (current_descr_statistics_y + range_mouseover) && yScale(y.descriptive_statistics[y_axis_key]) > (current_descr_statistics_y - range_mouseover));

    if (y_axis_key === key_data_type) {
        filtered_ = filtered.filter(y => yScale(y[y_axis_key]) < (current_descr_statistics_y + range_mouseover) && yScale(y[y_axis_key]) > (current_descr_statistics_y - range_mouseover));
    }

    this_._tippy.enable();

    if (filtered_.length > 1) {

        if (this_._tippy) {
            this_._tippy.disable();
        }

        overplotted_variables = filtered_;

        let i_x = 5;

        let number_of_rows = Math.ceil(filtered_.length / i_x);
        let number_of_cols = filtered_.length % i_x;

        if (filtered_.length >= i_x) {
            number_of_cols = i_x;
        }

        let svg = d3.select('#view').append('div').attr('id', 'overplotting_tooltip_div').style("opacity", 1)
            .attr("class", "tooltip")
            .style("background-color", "white")
            .style('position', 'absolute')
            .style("border", "solid")
            .style("border-width", "2px")
            .style("border-radius", "5px")
            .style('width', (number_of_cols * 20 + 15) + 'px')
            .style('height', (number_of_rows * 20 + 15) + 'px')
            .style("padding", "5px")
            .style("left", (d3.event.pageX - 10) + "px")
            .style("top", (d3.event.pageY - 28) + "px")
            .on("mouseover", function () {
                d3.select('#overplotting_tooltip_div').transition();
                for (let index_dims = 0; index_dims < overplotted_variables.length; index_dims++) {
                    d3.selectAll('#' + overplotted_variables[index_dims][key_id] + id_circle_dimension_ending).transition();
                }
            })
            .on("mouseout", function () {
                fade_out_overplotting_tooltip();
            })
            .append('svg').style('position', 'absolute')
            .attr('width', 100 + '%')
            .attr('height', 100 + '%')
            .attr('x', 0)
            .attr('y', 0);

        for (let index_dims = 0; index_dims < filtered_.length; index_dims++) {

            let circle = svg.append('circle').attr('cx', (index_dims % i_x) * 20 + 20 / 2)
                .attr('cy', Math.floor(index_dims / i_x) * 20 + 10)
                .attr('r', radius_dimension)
                .attr('fill', d3.select('#' + filtered_[index_dims].id + id_circle_dimension_ending).attr('fill'))
                .style('opacity', function (d) {
                    let opacity = 1 - filtered_[index_dims].descriptive_statistics.missing_values_percentage;
                    if (possible_correlations_bool) {
                        opacity = filtered_[index_dims][key_column_values].filter(x => x !== null).length / column_values_cleaned.find(col => col[key_id] === d[key_id])[key_column_values].length + min_opacity / 2;
                    }
                    opacity = opacity < min_opacity ? min_opacity : opacity;
                    opacity = opacity > 1 ? 1 : opacity;

                    return opacity;
                })
                .attr('data-tippy-content', function () {
                    return append_tooltip_radar_chart_and_heading(parent_div_id, filtered_[index_dims], false);
                })
                .on('mouseover', function () {
                    d3.select(this).style('opacity', 1);
                })
                .on("mouseout", function () {
                    d3.select(this).style('opacity', function (d) {
                        let opacity = 1 - filtered_[index_dims].descriptive_statistics.missing_values_percentage;
                        if (possible_correlations_bool) {
                            opacity = filtered_[index_dims][key_column_values].filter(x => x !== null).length / column_values_cleaned.find(col => col[key_id] === d[key_id])[key_column_values].length + min_opacity / 2;
                        }
                        opacity = opacity < min_opacity ? min_opacity : opacity;
                        opacity = opacity > 1 ? 1 : opacity;

                        return opacity;
                    });
                })
                .on('click', function (d) {

                    highlight_dimension(filtered_[index_dims][key_id]);
                    highlight_circles(filtered_[index_dims][key_id]);
                })

            tippy(circle.nodes(), {allowHTML: true})

            d3.selectAll('#' + filtered_[index_dims][key_id] + id_circle_dimension_ending).style('stroke', 'red').style('stroke-width', 5 + 'px')
        }
    }
}


function fade_out_overplotting_tooltip() {
    d3.select('#overplotting_tooltip_div').transition().duration(10).delay(3000).remove();

    let copy_overplotted_variables = JSON.parse(JSON.stringify(overplotted_variables));
    for (let index_dims = 0; index_dims < copy_overplotted_variables.length; index_dims++) {

        d3.selectAll('#' + copy_overplotted_variables[index_dims][key_id] + id_circle_dimension_ending).transition().duration(10).delay(3000).style('stroke-width', 0 + 'px');
    }
}