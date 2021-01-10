---
title: "Useful Identities of Computing Gradients"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post
---



## Useful Identities of Computing Gradients


$$
\frac{\partial}{\partial X} f(X)^T =(\frac{\partial f(X)}{\partial X})^T
$$

$$
\frac{\partial}{\partial X} tr(f(X)) =tr (\frac{\partial f(X)}{\partial X})
$$

$$
\frac{\partial}{\partial X} \det (f(X)) =\det(f(X)) tr(f(X) ^{-1}\frac{\partial f(X)}{\partial X})
$$

$$
\frac{\partial}{\partial X} f(X)^{-1} =-f(X)^{-1} \frac{\partial f(X)}{\partial X} f(X)^-1
$$

$$
\frac{\partial a^TX^{-1}b}{\partial X} = - (X^{-1})^Tab^T(X^{-1})^T
$$

$$
\frac{\partial x^T a}{\partial x} =a^T
$$

$$
\frac{\partial a^Tx}{\partial x} =a^T
$$

$$
\frac{\partial a^TXb}{\partial X} = a^Tb
$$

$$
\frac{\partial x^T B x}{\partial X} = x^T(B^T + B)
$$

$$
\frac{\partial}{\partial s}(x - As)^T W (x - As) = - 2(x - As)^T W A \text{ for symmetric matrix W}
$$

