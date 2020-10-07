/**
 * using backend for deviation computation
 * @param callback
 * @param active_patients
 */
function compute_deviations_new(callback, active_patients) {
    d3.selectAll('#' + id_view).style('pointer-events', 'none');

    $.ajax({
        url: "http://127.0.0.1:5000/compute_deviations_and_get_current_values/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify(active_patients)
    }).done(function(data) {

        column_values_filtered = JSON.parse(data)[0];
        column_values_filtered.sort((a,b) => a.descriptive_statistics[sort_parallel_coordinates_by] < b.descriptive_statistics[sort_parallel_coordinates_by] ? 1 : -1);

        let col_sort_originally = JSON.parse(JSON.stringify(column_values_grouped));
        let column_values_grouped_initial = JSON.parse(data)[0];
        //column_values_grouped.sort((a,b) => a.descriptive_statistics[sort_parallel_coordinates_by] < b.descriptive_statistics[sort_parallel_coordinates_by] ? 1 : -1);

        let result = [];
        col_sort_originally.forEach(function(key) {
            var found = false;
            column_values_grouped_initial = column_values_grouped_initial.filter(function(item) {
                if(!found && item.id == key.id) {
                    result.push(item);
                    found = true;
                    return false;
                } else
                    return true;
            })
        })

        column_values_grouped = result;

        d3.selectAll('#' + id_view).style('pointer-events', 'auto');

        callback(true);
    });
}