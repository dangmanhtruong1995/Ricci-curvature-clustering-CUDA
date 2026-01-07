#include <stdio.h>

void prefix_sum_exclusive(int *arr, int *result, int n) {
    result[0] = 0;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] + arr[i - 1];
    }
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    int result[n];

    prefix_sum_exclusive(arr, result, n);

    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    printf("\nExclusive prefix sum: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    return 0;
}
