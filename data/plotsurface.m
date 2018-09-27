clc;
clear all;
close all;

ptCloud = pcread('mesh.ply');
xyz = double(ptCloud.Location);

x = xyz(:,1);
y = xyz(:,2);
z = xyz(:,3);
tri = delaunay(x,y);
h = trisurf(tri, x, y, z);
axis vis3d