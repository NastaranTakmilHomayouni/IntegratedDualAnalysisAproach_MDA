/**
 * load data from csv
 * @param callback
 */
function load_csv(callback) {
    d3.selectAll('#' + id_view).style('pointer-events', 'none');

    $.ajax({
        url: "http://127.0.0.1:5000/load_csv/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({"message": "test"})
    }).done(function(data) {

        column_values_initially_with_missing_values = JSON.parse(data)[0];
        column_values_initially = JSON.parse(data)[0]; //format_datatypes_regarding_datatype(JSON.parse(JSON.stringify(column_values_right_from_data_without_formatting)));
        column_values_initially_no_missing_values = JSON.parse(data)[1];

        column_values_cleaned = JSON.parse(JSON.stringify(column_values_initially));
        column_values_filtered = JSON.parse(JSON.stringify(column_values_initially));

        let summed_missing = 0;
        column_values_cleaned.forEach(function (col) {
            summed_missing += col.descriptive_statistics[statistics_key__missing_values_percentage];
        });

        column_values_cleaned.forEach(function (column, index_col) {
            column.column_values.forEach(function (col_value, index_val) {
                if (!data_table_cleaned[index_val]) {
                    data_table_cleaned[index_val] = {};
                }
                data_table_cleaned[index_val][column.id] = col_value;
            })
        });

        column_values_cleaned_with_missing_values = JSON.parse(JSON.stringify(column_values_cleaned));
        column_values_cleaned_no_missing_values = JSON.parse(data)[1];

        column_values_cleaned_no_missing_values.forEach(function (column, index_col) {
            column.column_values.forEach(function (col_value, index_val) {
                if (!data_table_cleaned[index_val]) {
                    data_table_cleaned[index_val] = {};
                }
                data_table_cleaned[index_val][column.id] = col_value;
            })
        });

        column_values_filtered.sort((a,b) => a.descriptive_statistics[sort_parallel_coordinates_by] < b.descriptive_statistics[sort_parallel_coordinates_by] ? 1 : -1);

        column_values_filtered.forEach(function (dim) {
            // MDA!

            column_values_grouped.push(dim);

        });

        d3.selectAll('#' + id_view).style('pointer-events', 'auto');

        console.log(column_values_filtered);

        callback(true)
    });
}