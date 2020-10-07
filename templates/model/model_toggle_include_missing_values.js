/**
 * using backend for deviation computation
 * @param callback
 */
function toggle_include_missing_values(callback, active_patients) {

    if (include_missing_values_bool) {
        column_values_cleaned = column_values_cleaned_with_missing_values;
        column_values_initially = column_values_cleaned_with_missing_values;
    } else {
        column_values_cleaned = column_values_cleaned_no_missing_values;
        column_values_initially = column_values_initially_no_missing_values;

    }

    $.ajax({
        url: "http://127.0.0.1:5000/toggle_include_missing_values/",
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