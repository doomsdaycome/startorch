#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void quicksort(int32_t *a, int64_t l, int64_t r)
{
	int32_t pivot = a[l + (r - l) / 2];
	int64_t i = l;
	int64_t j = r;

	while (i <= j) {
		while (a[i] < pivot)
			i++;

		while (a[j] > pivot)
			j--;

		if (i <= j) {
			int32_t temp = a[i];

			a[i] = a[j];
			a[j] = temp;

			i++;
			j--;
		}
	}

	if (l < j)
		quicksort(a, l, j);

	if (i < r)
		quicksort(a, i, r);
}

int main(void)
{
	int32_t a[11] = { 1, 3, 5, 7, 4, 5, 2, 6, 7, 9, 8 };

	quicksort(a, 0, 10);

	for (int64_t i = 0; i < 11; i++)
		printf("%d ", a[i]);

	printf("\n");

	return EXIT_SUCCESS;
}
