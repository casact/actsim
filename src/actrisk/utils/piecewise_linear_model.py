import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PiecewiseLinearModel:
    def __init__(self, startpoint, breakpoints, slopes):
        """
        Initialize the Piecewise Linear Model.

        Parameters:
        - breakpoints: list of x values where the slope changes.
        - slopes: list of slopes between each pair of breakpoints.
        """
        if len(slopes) != len(breakpoints) - 1:
            raise ValueError("Number of slopes must be one less than number of breakpoints.")
        
        self.startpoint = startpoint
        self.breakpoints = np.array(breakpoints)
        self.slopes = np.array(slopes)
        self.intercepts = self._calculate_intercepts()

    def _calculate_intercepts(self):
        """
        Calculate intercepts based on breakpoints and slopes.
        """
        intercepts = [self.startpoint]  # Assume the curve starts at y=0 at the first breakpoint
        for i in range(1, len(self.breakpoints)):
            intercepts.append(intercepts[i-1] + self.slopes[i-1] * (self.breakpoints[i] - self.breakpoints[i-1]))
        return np.array(intercepts)

    def evaluate(self, x):
        """
        Evaluate the piecewise linear function at a vector of x values (vectorized).
        """
        # Initialize an array of zeros for y-values
        y = np.zeros_like(x, dtype=float)

        # Apply each slope segment piecewise
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i + 1])
            y[mask] = self.slopes[i] * (x[mask] - self.breakpoints[i]) + self.intercepts[i]

        # Handle the case where x is beyond the last breakpoint
        mask_last_segment = (x >= self.breakpoints[-2])
        y[mask_last_segment] = self.slopes[-1] * (x[mask_last_segment] - self.breakpoints[-2]) + self.intercepts[-2]

        return y

    def apply_to_column(self, col:pd.Series):
        col_pred = self.evaluate(col)
        return col_pred

'''
# Example usage:
breakpoints = [0, 1, 2, 3, 5]  # The x values where the slopes change
slopes = [2, -1, 0.5, 3]  # The slopes between the breakpoints

model = PiecewiseLinearModel(breakpoints, slopes)

# Create a sample DataFrame
df = pd.Series(np.linspace(0, 6, 100))

# Apply the piecewise function to the 'x' column
df_pred = model.apply_to_column(df)

# Plot the result
plt.plot(df, df_pred, label="Transformed X")
plt.xlabel('X')
plt.ylabel('Transformed X')
plt.title('Vectorized Piecewise Linear Transformation')
plt.grid(True)
plt.show()
'''